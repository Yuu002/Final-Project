from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import numpy as np
from indices import calculate_indices
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, MimeType, DataCollection

app = FastAPI()

# ===== CORS =====
origins = ["*"]  # สามารถเปลี่ยนเป็น domain frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== path dynamic =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MLP_MODEL_PATH = os.path.join(BASE_DIR, "mlp_model.pkl")
SCALER_MLP_X_PATH = os.path.join(BASE_DIR, "scaler_mlp_X.pkl")
SCALER_MLP_Y_PATH = os.path.join(BASE_DIR, "scaler_mlp_y.pkl")
SVR_AGB_MODEL_PATH = os.path.join(BASE_DIR, "svr_agb_model.pkl")
SCALER_SVR_X_PATH = os.path.join(BASE_DIR, "scaler_svr_X.pkl")

# ===== load models =====
mlp_model = joblib.load(MLP_MODEL_PATH)
scaler_mlp_X = joblib.load(SCALER_MLP_X_PATH)
scaler_mlp_y = joblib.load(SCALER_MLP_Y_PATH)
svr_agb_model = joblib.load(SVR_AGB_MODEL_PATH)
scaler_svr_X = joblib.load(SCALER_SVR_X_PATH)

print("BASE_DIR =", BASE_DIR)
print("Files in BASE_DIR:", os.listdir(BASE_DIR))

# ===== Schema =====
class CoordInput(BaseModel):
    lat: float
    lon: float

# ===== ฟังก์ชัน scale bands 40-78 =====
def scale_bands_to_range(bands, min_val=40, max_val=78):
    bands = np.array(bands, dtype=float)
    if bands.max() - bands.min() == 0:
        return np.full_like(bands, (min_val+max_val)//2, dtype=int)
    bands_norm = (bands - bands.min()) / (bands.max() - bands.min())
    bands_scaled = bands_norm * (max_val - min_val) + min_val
    return np.round(bands_scaled).astype(int)

# ===== ดึง Sentinel bands จาก lat/lon =====
def get_sentinel_bands(lat, lon):
    config = SHConfig()
    # ใช้ environment variable หรือใส่ค่าโดยตรง
    config.sh_client_id = os.getenv("5fe79f43-3c2d-4cf6-9184-478e247504b7")
    config.sh_client_secret = os.getenv("uYQ6E6NvUDucLx0yRqz4zrUyeuWcX09U")

    print(f"Fetching Sentinel bands for lat={lat}, lon={lon}")

    bbox = BBox(bbox=[lon, lat, lon, lat], crs=CRS.WGS84)
    
    request = SentinelHubRequest(
        evalscript="""
            //VERSION=3
            function setup() {
                return { input: ["B02","B03","B04","B08"], output: { bands: 4 } };
            }
            function evaluatePixel(sample) {
                return [sample.B02*100, sample.B03*100, sample.B04*100, sample.B08*100];
            }
        """,
        input_data=[SentinelHubRequest.input_data(DataCollection.SENTINEL2_L2A)],
        responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
        bbox=bbox,
        size=(1,1),
        config=config
    )
    
    data = request.get_data()[0][0][0]  # shape (1,1,4)
    return np.round(data).astype(int).tolist()

# ===== Endpoint: ส่ง lat/lon → pipeline full =====
@app.post("/predict_from_coords")
def predict_from_coords(data: CoordInput):
    sentinel_bands = get_sentinel_bands(data.lat, data.lon)
    sentinel_bands_int = scale_bands_to_range(sentinel_bands, 40, 78)
    
    X_scaled = scaler_mlp_X.transform([sentinel_bands_int])
    ground_bands_scaled = mlp_model.predict(X_scaled)
    ground_bands = scaler_mlp_y.inverse_transform(ground_bands_scaled)[0]
    ground_bands_int = scale_bands_to_range(ground_bands, 40, 78)
    blue, green, red, nir = ground_bands_int
    
    indices = calculate_indices(red=red, nir=nir, blue=blue, green=green)
    indices_array = np.array([[indices["NDVI"], indices["TNDVI"], indices["SR"],
                               indices["SAVI"], indices["MSAVI2"]]])
    indices_scaled = scaler_svr_X.transform(indices_array)
    agb_pred = svr_agb_model.predict(indices_scaled)[0]
    
    print(f"Sentinel bands: {sentinel_bands_int}, AGB: {agb_pred:.2f}")

    return {
        "sentinel_bands": sentinel_bands_int,
        "ground_bands": {"Blue": int(blue), "Green": int(green), "Red": int(red), "NIR": int(nir)},
        "indices": indices,
        "AGB_prediction": float(agb_pred)
    }

# ===== run uvicorn =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)