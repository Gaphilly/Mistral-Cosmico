from flask import Flask, request, jsonify, send_from_directory
import os
import datetime
import requests
import xarray as xr
import numpy as np
import re

# --- Flask app ---
app = Flask(__name__, static_folder="static")

# --- Configuration ---
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

NETRC_PATH = "/etc/secrets/.netrc"
os.environ["NETRC"] = NETRC_PATH

# --- Base NASA URL ---
BASE_URL = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2SDNXSLV.5.12.4"

# --- Utilities ---
def list_month_files(year, month):
    url = f"{BASE_URL}/{year}/{month:02d}/"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            files = re.findall(r'href="(MERRA2_\d+\.statD_2d_slv_Nx\.\d+\.nc4)"', r.text)
            return files
        else:
            return []
    except:
        return []

def find_file_for_date(year, month, day):
    date_tag = f"{year}{month:02d}{day:02d}"
    month_files = list_month_files(year, month)
    for f in month_files:
        if date_tag in f:
            return f
    return None

def download_file(url):
    filename = os.path.join(CACHE_DIR, url.split("/")[-1])
    if os.path.exists(filename):
        return filename
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return filename
        return None
    except:
        return None

def extract_daily_averages(filename, lat_center, lon_center):
    ds = xr.open_dataset(filename)
    subset = ds.sel(
        lat=slice(lat_center - 0.5, lat_center + 0.5),
        lon=slice(lon_center - 0.5, lon_center + 0.5),
    )
    t2m_avg = float(subset["T2MMEAN"].mean() - 273.15)
    prectot_avg = float(subset["TPRECMAX"].mean() * 86400)  # mm/day approx.
    ds.close()
    return t2m_avg, prectot_avg

def compute_historical_averages(day, month, lat, lon, years_back=15):
    today = datetime.date.today()
    t2m_vals = []
    rainfall_occurrences = []

    for y in range(today.year - years_back, today.year):
        fname = find_file_for_date(y, month, day)
        if fname:
            url = f"{BASE_URL}/{y}/{month:02d}/{fname}"
            f = download_file(url)
            if f:
                t2m, prec = extract_daily_averages(f, lat, lon)
                t2m_vals.append(t2m)
                rainfall_occurrences.append(1 if prec > 2 else 0)

    avg_t2m = np.mean(t2m_vals) if t2m_vals else None
    rainfall_freq_percent = int(np.mean(rainfall_occurrences) * 100) if rainfall_occurrences else None

    return avg_t2m, rainfall_freq_percent

# --- Routes ---
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/style.css")
def serve_style():
    return send_from_directory(app.static_folder, "style.css")

@app.route("/climate_stats")
def climate_stats():
    day = request.args.get("day", type=int)
    month = request.args.get("month", type=int)
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if day is None or month is None or lat is None or lon is None:
        return jsonify({"error": "Missing parameters. Provide day, month, lat, lon."}), 400

    avg_t2m, rainfall_freq_percent = compute_historical_averages(day, month, lat, lon)
    return jsonify({
        "day": day,
        "month": month,
        "lat": lat,
        "lon": lon,
        "avg_temp_C": avg_t2m,
        "rainfall_gt_2mm_freq_percent": rainfall_freq_percent
    })

# --- Entry point ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
