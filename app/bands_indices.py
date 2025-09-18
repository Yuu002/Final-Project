# bands_predict.py
import rasterio
import numpy as np
from pyproj import Transformer
import joblib

def compute_bands_and_indices_sklearn(raster_path, lat, lon,
                                      model_path="mlp_model.pkl",
                                      scaler_X_path="scaler_X.save",
                                      scaler_y_path="scaler_y.save"):
    """
    ทำนายค่า Blue, Green, Red, NIR ด้วย sklearn MLP
    จากนั้นคำนวณดัชนี NDVI, TNDVI, SR, SAVI, MSAVI2
    """
    # --- Load model + scalers ---
    model = joblib.load(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # --- Open raster ---
    with rasterio.open(raster_path) as src:
        crs = src.crs
        transform = src.transform
        data = src.read([1,2,3,4])
        nodata = src.nodata

    # --- lat/lon -> pixel ---
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_center, y_center = transformer.transform(lon, lat)
    col_center, row_center = ~transform * (x_center, y_center)
    col_center = int(round(col_center))
    row_center = int(round(row_center))

    # --- 1x1 pixel window ---
    window = data[:, row_center:row_center+1, col_center:col_center+1]

    # --- Prepare input for model ---
    band_values = []
    for i in range(4):
        band_data = window[i]
        if nodata is not None:
            band_data = band_data[band_data != nodata]
        mean_val = float(np.mean(band_data)) if band_data.size > 0 else 0
        band_values.append(mean_val)

    X_scaled = scaler_X.transform([band_values])
    y_scaled_pred = model.predict(X_scaled)
    if y_scaled_pred.ndim == 1:
        y_scaled_pred = y_scaled_pred.reshape(1,-1)
    blue, green, red, nir = scaler_y.inverse_transform(y_scaled_pred)[0]

    bands_dict = {"Blue": blue, "Green": green, "Red": red, "NIR": nir}

    # --- Compute vegetation indices ---
    L = 0.5
    eps = 1e-6
    NDVI = (nir - red) / (nir + red + eps)
    TNDVI = np.sqrt(NDVI + L)
    SR = nir / (red + eps)
    SAVI = ((1+L)*(nir - red)) / (nir + red + L + eps)
    MSAVI2 = (2*nir + 1 - np.sqrt((2*nir +1)**2 - 8*(nir - red))) / 2

    indices_dict = {
        "NDVI": NDVI,
        "TNDVI": TNDVI,
        "SR": SR,
        "SAVI": SAVI,
        "MSAVI2": MSAVI2
    }

    return bands_dict, indices_dict