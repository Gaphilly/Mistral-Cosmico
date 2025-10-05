from flask import Flask, request, jsonify, send_from_directory
import os
import datetime
import requests
import xarray as xr
import numpy as np
import re

app = Flask(__name__, static_folder="static")

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Use .netrc authentication via environment variable (recommended by NASA)
os.environ["NETRC"] = os.getenv("NETRC_PATH", "/etc/secrets/.netrc")

BASE_URL = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2SDNXSLV.5.12.4"

def list_month_files(year, month):
    url = f"{BASE_URL}/{year}/{month:02d}/"
    print(f"[INFO] Listing files at {url}")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            files = re.findall(r'href="(MERRA2_\\d+\\.statD_2d_slv_Nx\\.\\d+\\.nc4)"', r.text)
            print(f"[INFO] Found {len(files)} files for {year}-{month:02d}")
            return files
        else:
            print(f"[ERROR] Directory list failed ({r.status_code}) at {url}")
            return []
    except Exception as e:
        print(f"[ERROR] Listing failed for {url}: {e}")
        return []

def find_file_for_date(year, month, day):
    date_tag = f"{year}{month:02d}{day:02d}"
    month_files = list_month_files(year, month)
    for f in month_files:
        if date_tag in f:
            print(f"[INFO] Found file: {f}")
            return f
    print(f"[WARN] No file found for {date_tag}")
    return None

def generate_urls_for_past_15_years(month, day):
    """Generate URLs for this day/month over the last 15 years."""
    current_year = datetime.date.today().year
    urls = []
    for year in range(current_year - 15, current_year):
        fname = find_file_for_date(year, month, day)
        if fname:
            urls.append(f"{BASE_URL}/{year}/{month:02d}/{fname}")
    return urls

def download_file(url):
    filename = os.path.join(CACHE_DIR, url.split("/")[-1])
    if os.path.exists(filename):
        print(f"[CACHE] Using {filename}")
        return filename
    print(f"[DL] {url}")
    try:
        response = requests.get(url, stream=True, timeout=30, auth=(os.getenv("NASA_USER"), os.getenv("NASA_PASS")))
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"[OK] Downloaded {filename}")
            return filename
        else:
            print(f"[ERROR] Failed ({response.status_code}) {url}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception downloading {url}: {e}")
        return None

def extract_precip(filename, lat, lon):
    """Extract total precipitation (mm/day) over a 1°x1° box."""
    ds = xr.open_dataset(filename)
    subset = ds.sel(lat=slice(lat - 0.5, lat + 0.5), lon=slice(lon - 0.5, lon + 0.5))
    prectot = float(subset["TPRECMAX"].mean() * 86400)  # mm/day approx
    ds.close()
    return prectot

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/style.css")
def serve_style():
    return send_from_directory(app.static_folder, "style.css")

@app.route("/precip_freq")
def precip_frequency():
    """Return fraction of past 15 years where daily rain > 2mm for given day/month."""
    month = request.args.get("month", type=int)
    day = request.args.get("day", type=int)
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if not all([month, day, lat, lon]):
        return jsonify({"error": "Missing parameters: month, day, lat, lon"}), 400

    urls = generate_urls_for_past_15_years(month, day)
    count_above = 0
    total = 0

    for url in urls:
        f = download_file(url)
        if f:
            total += 1
            prec = extract_precip(f, lat, lon)
            if prec > 2:
                count_above += 1
            print(f"[DATA] {url.split('/')[-1]} => {prec:.2f} mm")

    if total == 0:
        return jsonify({"error": "No data available"}), 404

    freq = count_above / total
    return jsonify({
        "month": month,
        "day": day,
        "lat": lat,
        "lon": lon,
        "years_checked": total,
        "fraction_rain_above_2mm": round(freq, 3)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
