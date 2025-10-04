from flask import Flask, request, jsonify, send_from_directory
import os
import datetime
import requests
import xarray as xr
import numpy as np

app = Flask(__name__, static_folder="static")

# Cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
# NASA Earthdata credentials from environment variables
USERNAME = os.environ.get("NASA_USER")
PASSWORD = os.environ.get("NASA_PASS")

if USERNAME is None or PASSWORD is None:
    raise ValueError("NASA_USER and NASA_PASS environment variables must be set!")

# Base URL for MERRA-2 daily collection
BASE_URL = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2SDNXSLV.5.12.4"

def generate_urls(center_date_str):
    """Generate URLs for a 5-day window centered around the given date."""
    center_date = datetime.datetime.strptime(center_date_str, "%Y-%m-%d").date()
    start_date = center_date - datetime.timedelta(days=2)
    end_date = center_date + datetime.timedelta(days=2)
    urls = []
    for n in range((end_date - start_date).days + 1):
        date = start_date + datetime.timedelta(days=n)
        year, month, day = date.year, f"{date.month:02d}", f"{date.day:02d}"
        filename = f"MERRA2_400.statD_2d_slv_Nx.{year}{month}{day}.nc4"
        urls.append(f"{BASE_URL}/{year}/{month}/{filename}")
    return urls

def download_file(url):
    """Download a file and cache it locally, printing detailed errors on failure."""
    filename = os.path.join(CACHE_DIR, url.split("/")[-1])
    if os.path.exists(filename):
        print(f"[INFO] Using cached file: {filename}")
        return filename

    print(f"[INFO] Downloading {url} ...")
    try:
        response = requests.get(url, stream=True, timeout=30, auth=("gaphilly", "Bbroz678@lafase")) # type: ignore
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"[INFO] Downloaded successfully: {filename}")
            return filename
        else:
            print(f"[ERROR] Failed to download {url}")
            print(f"  Status code: {response.status_code}")
            print(f"  Response headers: {response.headers}")
            print(f"  Response text (first 500 chars): {response.text[:500]}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed for {url}")
        print(f"  Exception: {e}")
        return None

def extract_daily_averages(filename, lat_center, lon_center):
    """Extract daily T2M and precipitation averages over a 1°x1° square."""
    ds = xr.open_dataset(filename)
    subset = ds.sel(
        lat=slice(lat_center-0.5, lat_center+0.5),
        lon=slice(lon_center-0.5, lon_center+0.5)
    )

    # Temperature in Celsius
    t2m_avg = float(subset["T2MMEAN"].mean() - 273.15)

    # Precipitation in mm (daily maximum as proxy)
    prectot_avg = float(subset["TPRECMAX"].mean())

    ds.close()
    return t2m_avg, prectot_avg

def compute_averages(date_str, lat, lon):
    urls = generate_urls(date_str)
    t2m_vals, prec_vals = [], []

    for url in urls:
        f = download_file(url)
        if f:
            t2m, prec = extract_daily_averages(f, lat, lon)
            t2m_vals.append(t2m)
            prec_vals.append(prec)

    avg_t2m = np.mean(t2m_vals) if t2m_vals else None
    avg_prec = np.mean(prec_vals) if prec_vals else None

    return avg_t2m, avg_prec

# --- Flask Endpoints ---

@app.route("/temp_avg")
def temp_avg():
    date = request.args.get("date")
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if not date or lat is None or lon is None:
        return jsonify({"error": "Missing parameters. Provide date, lat, lon."}), 400

    avg_t2m, _ = compute_averages(date, lat, lon)
    return jsonify({"date": date, "lat": lat, "lon": lon, "temp_avg_C": avg_t2m})

@app.route("/precip_avg")
def precip_avg():
    date = request.args.get("date")
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if not date or lat is None or lon is None:
        return jsonify({"error": "Missing parameters. Provide date, lat, lon."}), 400

    _, avg_prec = compute_averages(date, lat, lon)
    return jsonify({"date": date, "lat": lat, "lon": lon, "precip_avg_mm": avg_prec})

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")