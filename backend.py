from flask import Flask, request, jsonify, send_from_directory
import os
import datetime
import requests
import xarray as xr
import numpy as np
import re
import sys
import json
import urllib3
import certifi
from time import sleep
import getpass
from urllib.request import HTTPPasswordMgrWithDefaultRealm, HTTPBasicAuthHandler, build_opener, install_opener, HTTPCookieProcessor, Request
from http.cookiejar import CookieJar

# --- Flask app ---
app = Flask(__name__, static_folder="static")

# --- Configuration ---
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

NETRC_PATH = ".netrc"
os.environ["NETRC"] = NETRC_PATH

BASE_URL = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2SDNXSLV.5.12.4"

# ------------------------------
# Logging
# ------------------------------
def log(msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")

# ------------------------------
# Existing climate functions
# ------------------------------
def list_month_files(year, month):
    url = f"{BASE_URL}/{year}/{month:02d}/"
    log(f"Listing files at {url}")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            files = re.findall(r'href="(MERRA2_\d+\.statD_2d_slv_Nx\.\d+\.nc4)"', r.text)
            log(f"Found {len(files)} files for {year}-{month:02d}")
            return files
        else:
            log(f"[WARN] Failed to list {url}, status {r.status_code}")
            return []
    except Exception as e:
        log(f"[ERROR] Exception listing month files: {e}")
        return []

def find_file_for_date(year, month, day):
    date_tag = f"{year}{month:02d}{day:02d}"
    month_files = list_month_files(year, month)
    for f in month_files:
        if date_tag in f:
            log(f"Found file for {date_tag}: {f}")
            return f
    log(f"[WARN] No file found for {date_tag}")
    return None

def download_file(url):
    filename = os.path.join(CACHE_DIR, url.split("/")[-1])
    if os.path.exists(filename):
        log(f"Using cached file: {filename}")
        return filename
    try:
        log(f"Downloading {url} ...")
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            log(f"Downloaded successfully: {filename}")
            return filename
        else:
            log(f"[ERROR] Failed to download {url}, status {response.status_code}")
            return None
    except Exception as e:
        log(f"[ERROR] Download error for {url}: {e}")
        return None

def extract_daily_averages(filename, lat_center, lon_center):
    log(f"Extracting data from {filename} at lat={lat_center}, lon={lon_center}")
    ds = xr.open_dataset(filename)
    subset = ds.sel(
        lat=slice(lat_center - 0.5, lat_center + 0.5),
        lon=slice(lon_center - 0.5, lon_center + 0.5),
    )
    t2m_avg = float(subset["T2MMEAN"].mean() - 273.15)
    prectot_avg = float(subset["TPRECMAX"].mean() * 86400)  # mm/day approx.
    ds.close()
    log(f"Extracted: T2M={t2m_avg:.2f}°C, Precip={prectot_avg:.2f} mm/day")
    return t2m_avg, prectot_avg

def compute_historical_stats(day, month, lat, lon, years_back=15):
    today = datetime.date.today()
    t2m_vals = []
    rainfall_occurrences = []
    heat_occurrences = []

    log(f"Computing historical stats for {day:02d}/{month:02d} over {years_back} years")
    for y in range(today.year - years_back, today.year):
        log(f"Processing year {y}")
        fname = find_file_for_date(y, month, day)
        if fname:
            url = f"{BASE_URL}/{y}/{month:02d}/{fname}"
            f = download_file(url)
            if f:
                t2m, prec = extract_daily_averages(f, lat, lon)
                t2m_vals.append(t2m)
                rainfall_occurrences.append(1 if prec > 2 else 0)
                heat_occurrences.append(1 if t2m > 35 else 0)
        else:
            log(f"No file for {y}-{month:02d}-{day:02d}")

    avg_t2m = np.mean(t2m_vals) if t2m_vals else None
    rainfall_freq_percent = int(np.mean(rainfall_occurrences) * 100) if rainfall_occurrences else None
    heat_freq_percent = int(np.mean(heat_occurrences) * 100) if heat_occurrences else None

    log(f"Final averages: Temp={avg_t2m}, Rain freq={rainfall_freq_percent}%, Heat freq={heat_freq_percent}%")
    return avg_t2m, rainfall_freq_percent, heat_freq_percent

# ------------------------------
# Wind speed function (EarthData subset)
# ------------------------------
http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
EARTHDATA_SUBSET_URL = 'https://disc.gsfc.nasa.gov/service/subset/jsonwsp'

def get_http_data(request):
    hdrs = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    data = json.dumps(request)
    r = http.request('POST', EARTHDATA_SUBSET_URL, body=data, headers=hdrs)
    response = json.loads(r.data)
    if response['type'] == 'jsonwsp/fault':
        log(f"API Error: faulty {response['methodname']} request")
        return None
    return response

def compute_wind_speed_stats(lat_center, lon_center, years_back=15):
    product = 'M2T1NXSLV_5.12.4'
    varNames = ['U10M', 'V10M']
    diurnalAggregation = '1'
    interp = 'remapbil'
    destGrid = 'cfsr0.5a'

    username = os.environ.get("EARTHDATA_USER") or input("EarthData userid: ")
    password = os.environ.get("EARTHDATA_PASS") or getpass.getpass("EarthData password: ")

    # Setup auth for file download
    password_manager = HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)
    cookie_jar = CookieJar()
    opener = build_opener(HTTPBasicAuthHandler(password_manager), HTTPCookieProcessor(cookie_jar))
    install_opener(opener)

    # Track how many days wind > 15 m/s
    high_wind_occurrences = []

    today = datetime.date.today()
    for y in range(today.year - years_back, today.year):
        begTime = f"{y}-01-01"
        endTime = f"{y}-12-31"

        subset_request = {
            'methodname': 'subset',
            'type': 'jsonwsp/request',
            'version': '1.0',
            'args': {
                'role': 'subset',
                'start': begTime,
                'end': endTime,
                'box': [lon_center-0.5, lat_center-0.5, lon_center+0.5, lat_center+0.5],
                'crop': True,
                'diurnalAggregation': diurnalAggregation,
                'mapping': interp,
                'grid': destGrid,
                'data': [{'datasetId': product, 'variable': var} for var in varNames]
            }
        }

        response = get_http_data(subset_request)
        if response is None:
            continue
        jobId = response['result']['jobId']

        # Monitor job
        status_request = {'methodname': 'GetStatus', 'version': '1.0', 'type': 'jsonwsp/request', 'args': {'jobId': jobId}}
        status_resp = response
        while status_resp['result']['Status'] in ['Accepted', 'Running']: # type: ignore
            sleep(5)
            status_resp = get_http_data(status_request)

        if status_resp['result']['Status'] != 'Succeeded': # type: ignore
            log(f"[WARN] Wind job failed for year {y}")
            continue

        # Get result URLs
        results_request = {'methodname': 'GetResult', 'version': '1.0', 'type': 'jsonwsp/request',
                           'args': {'jobId': jobId, 'count': 20, 'startIndex': 0}}
        results = []
        count = 0
        response_result = get_http_data(results_request)
        count += response_result['result']['itemsPerPage'] # type: ignore
        results.extend(response_result['result']['items']) # type: ignore
        total = response_result['result']['totalResults'] # type: ignore

        while count < total:
            results_request['args']['startIndex'] += 20
            response_result = get_http_data(results_request)
            count += response_result['result']['itemsPerPage'] # type: ignore
            results.extend(response_result['result']['items']) # type: ignore

        urls = [item for item in results if 'start' in item and 'end' in item]
        filenames = []
        for item in urls:
            URL = item['link']
            DataRequest = Request(URL)
            DataResponse = opener.open(DataRequest)
            DataBody = DataResponse.read()
            file_name = os.path.join(CACHE_DIR, item['label'])
            with open(file_name, 'wb') as f:
                f.write(DataBody)
            filenames.append(file_name)

        for file in filenames:
            ds = xr.open_dataset(file)
            u = ds['U10M'].values
            v = ds['V10M'].values
            wind = np.sqrt(u**2 + v**2)
            # Daily max wind over the tile
            daily_max = np.max(wind, axis=0)
            high_wind_occurrences.append(1 if daily_max > 15 else 0)
            ds.close()

    if high_wind_occurrences:
        # Return frequency percent over 15 years
        return int(np.mean(high_wind_occurrences) * 100)
    return None


# --- Routes ---
@app.route("/")
def serve_index():
    log("Serving index.html")
    return send_from_directory(app.static_folder, "index.html")  # type: ignore

@app.route("/style.css")
def serve_style():
    log("Serving style.css")
    return send_from_directory(app.static_folder, "style.css")  # type: ignore

@app.route("/logo.jpg")
def serve_logo():
    log("Serving logo.jpg")
    return send_from_directory(app.static_folder, "logo.jpg")  # type: ignore

@app.route("/climate_stats")
def climate_stats():
    day = request.args.get("day", type=int)
    month = request.args.get("month", type=int)
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    log(f"Received request: day={day}, month={month}, lat={lat}, lon={lon}")
    if day is None or month is None or lat is None or lon is None:
        log("[ERROR] Missing parameters")
        return jsonify({"error": "Missing parameters. Provide day, month, lat, lon."}), 400

    # Existing stats
    avg_t2m, rainfall_freq_percent, heat_freq_percent = compute_historical_stats(day, month, lat, lon)

    # Wind speed stats
    try:
        avg_wind_speed = compute_wind_speed_stats(lat, lon, years_back=15)
    except Exception as e:
        log(f"[ERROR] Wind computation failed: {e}")
        avg_wind_speed = None

    # Decide label for precipitation
    if avg_t2m is not None and avg_t2m < -5:
        precip_label = "snow_hail_freq_percent"
        precip_value = rainfall_freq_percent
    else:
        precip_label = "rainfall_gt_2mm_freq_percent"
        precip_value = rainfall_freq_percent

    log(f"Returning data with precipitation label {precip_label}")
    return jsonify({
        "day": day,
        "month": month,
        "lat": lat,
        "lon": lon,
        "avg_temp_C": avg_t2m,
        precip_label: precip_value,
        "high_heat_freq_percent": heat_freq_percent,
        "avg_10m_wind_speed_m_s": avg_wind_speed
    })

# --- Entry point ---
if __name__ == "__main__":
    log("Starting Flask server on 0.0.0.0:10000")
    app.run(host="0.0.0.0", port=10000, debug=True)
