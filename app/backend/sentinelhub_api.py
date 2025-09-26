import requests
import os

# ต้องเซ็ต ENV (CLIENT_ID, CLIENT_SECRET) ใน Render หรือ .env
CLIENT_ID = os.getenv("SENTINEL_CLIENT_ID")
CLIENT_SECRET = os.getenv("SENTINEL_CLIENT_SECRET")

def get_access_token():
    url = "https://services.sentinel-hub.com/oauth/token"
    headers = {"content-type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    r = requests.post(url, headers=headers, data=data)
    r.raise_for_status()
    return r.json()["access_token"]

def get_bands(lat, lon, date="2019-02-15", size=20):
    """
    ดึงค่า reflectance B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
    """
    token = get_access_token()
    url = "https://services.sentinel-hub.com/api/v1/process"
    headers = {"Authorization": f"Bearer {token}"}
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B02","B03","B04","B08"],
        output: { bands: 4 }
      }
    }
    function evaluatePixel(sample) {
      return [sample.B02, sample.B03, sample.B04, sample.B08];
    }
    """
    payload = {
        "input": {
            "bounds": {
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                }
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date}T00:00:00Z",
                            "to": f"{date}T23:59:59Z",
                        },
                    },
                }
            ],
        },
        "aggregation": {
            "timeRange": {
                "from": f"{date}T00:00:00Z",
                "to": f"{date}T23:59:59Z",
            },
            "aggregationInterval": {"of": "P1D"},
            "resx": size,
            "resy": size,
        },
        "evalscript": evalscript,
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    arr = r.json()["data"][0]["bands"]
    return arr  # [Blue, Green, Red, NIR]