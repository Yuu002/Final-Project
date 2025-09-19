# app.py
import streamlit as st
import pandas as pd
import joblib
import tempfile
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
from bands_indices import compute_bands_and_indices_sklearn

st.set_page_config(page_title="AGB Predictor", layout="wide")
st.title("🌿 AGB Prediction from Satellite Image")

# --- Load AGB model ---
# --- Load AGB model ---
from pathlib import Path

try:
    model_path = Path(__file__).parent / "svr_agb_model.pkl"
    model_agb = joblib.load(model_path)
except Exception as e:
    st.error(f"Cannot load AGB model: {e}")
    model_agb = None

# --- Two columns ---
col_left, col_right = st.columns([1,1])

# --- Upload raster ---
with col_left:
    uploaded_file = st.file_uploader("Upload Satellite Image (TIFF)", type=["tif","tiff"])
    raster_path = None
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            tmp_file.write(uploaded_file.read())
            raster_path = tmp_file.name

# --- User inputs ---
with col_right:
    st.subheader("📋 Input Parameters")
    col1, col2 = st.columns(2)
    with col1:
        lat_str = st.text_input("Latitude (N)", value="0.0")
        lon_str = st.text_input("Longitude (E)", value="0.0")
    with col2:
        width_m = st.number_input("Plot width (m)", value=10.0)
        height_m = st.number_input("Plot height (m)", value=12.0)

    # --- Convert to float ---
    try:
        lat = float(lat_str)
    except:
        lat = 0.0
    try:
        lon = float(lon_str)
    except:
        lon = 0.0

# --- Show raster preview ---
with col_left:
    if raster_path:
        try:
            with rasterio.open(raster_path) as src:
                bands_to_show = min(3, src.count)
                img_array = src.read(list(range(1, bands_to_show+1)))
                img_array = np.transpose(img_array, (1,2,0))
                if img_array.dtype != np.uint8:
                    img_array = ((img_array - np.nanmin(img_array)) /
                                 np.nanmax(img_array - np.nanmin(img_array)) * 255).astype(np.uint8)

                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x_pix, y_pix = transformer.transform(lon, lat)
                row, col = src.index(x_pix, y_pix)

                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(img_array)
                ax.scatter([col], [row], color='red', s=50)
                ax.axis('off')
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Cannot display image: {e}")

# --- Predict button ---
with col_right:
    if st.button("Predict AGB"):
        if not raster_path:
            st.error("Please upload a raster image first")
        elif model_agb is None:
            st.error("AGB model not loaded")
        else:
            try:
                bands_dict, indices_dict = compute_bands_and_indices_sklearn(
                    raster_path, lat, lon,
                    model_path="mlp_model.pkl",
                    scaler_X_path="scaler_X.save",
                    scaler_y_path="scaler_y.save"
                )

                st.subheader("📊 Predicted Bands")
                st.table(pd.DataFrame(list(bands_dict.items()), columns=["Band","Value"]))

                st.subheader("📈 Vegetation Indices")
                st.table(pd.DataFrame(list(indices_dict.items()), columns=["Index","Value"]))

                # --- Predict AGB using 5 indices ---
                feature_order = ["NDVI","TNDVI","SR","SAVI","MSAVI2"]
                X_agb = [indices_dict[f] for f in feature_order]
                agb_pred = model_agb.predict([X_agb])[0]

                st.subheader("🌱 Predicted AGB (ton/ha)")
                st.markdown(f"<h2 style='color:green;font-size:30px;'>{agb_pred:.3f}</h2>", unsafe_allow_html=True)

            except Exception as e:

                st.error(f"Error predicting AGB: {e}")

