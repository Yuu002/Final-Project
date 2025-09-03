# app.py
import streamlit as st
import pandas as pd
import joblib
import tempfile
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
from bands_indices import compute_bands_and_indices

st.set_page_config(page_title="AGB Predictor", layout="wide")
st.title("🌿 AGB Prediction from Satellite Image")

# --- Create two columns: left for upload+image, right for info ---
col_left, col_right = st.columns([1, 1])

# --- Upload raster + show image preview with point overlay on left ---
with col_left:
    uploaded_file = st.file_uploader("Upload Satellite Image (TIFF)", type=["tif","tiff"])
    raster_path = None
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            tmp_file.write(uploaded_file.read())
            raster_path = tmp_file.name

# --- User inputs in two columns on right ---
with col_right:
    st.subheader("📋 Input Parameters")
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        lat_str = st.text_input("Latitude (N)", value="0.0")
        lon_str = st.text_input("Longitude (E)", value="0.0")
    with col_input2:
        width_m = st.number_input("Plot width (m)", value=10.0)
        height_m = st.number_input("Plot height (m)", value=12.0)

    # --- Convert inputs to float ---
    try:
        lat = float(lat_str)
    except:
        lat = 0.0
    try:
        lon = float(lon_str)
    except:
        lon = 0.0

    # --- Load model ---
    try:
        model = joblib.load("svr_agb_model.pkl")
    except:
        st.error("Cannot load model svr_agb_model.pkl")
        model = None

# --- Show image with overlay point on left ---
with col_left:
    if raster_path is not None:
        try:
            with rasterio.open(raster_path) as src:
                bands_to_show = min(3, src.count)
                img_array = src.read(list(range(1, bands_to_show+1)))
                img_array = np.transpose(img_array, (1,2,0))
                if img_array.dtype != np.uint8:
                    img_array = ((img_array - np.nanmin(img_array)) / 
                                 np.nanmax(img_array - np.nanmin(img_array)) * 255).astype(np.uint8)

                # แปลงพิกัด Lat/Lon -> Pixel ของภาพ
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x_img, y_img = transformer.transform(lon, lat)
                row, col = src.index(x_img, y_img)

                # วาดภาพด้วย matplotlib และ overlay จุด
                fig, ax = plt.subplots(figsize=(6,6))
                ax.imshow(img_array)
                ax.scatter([col], [row], color='red', s=50, marker='o')  # จุดพิกัด
                ax.set_title("Uploaded Image Preview with selected point")
                ax.axis('off')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Cannot display image: {e}")
    else:
        st.info("No image uploaded yet.")

# --- Predict button and results ---
with col_right:
    if st.button("Predict AGB"):
        if raster_path is None:
            st.error("Please upload a raster image first")
        elif model is None:
            st.error("Model not loaded")
        else:
            try:
                bands_dict, indices_dict = compute_bands_and_indices(raster_path, lat, lon, width_m, height_m)

                # Bands table
                st.subheader("📊 Bands (automatic)")
                bands_df = pd.DataFrame(list(bands_dict.items()), columns=["Band", "Mean Value"])
                st.table(bands_df)

                # Indices table
                st.subheader("📈 Vegetation Indices")
                indices_df = pd.DataFrame(list(indices_dict.items()), columns=["Index", "Value"])
                st.table(indices_df)

                # Predict AGB
                feature_order = ["NDVI","TNDVI","SR","SAVI","MSAVI2"]
                X = [indices_dict[f] for f in feature_order]
                agb_pred = model.predict([X])[0]

                st.subheader("🌱 Predicted AGB (ton/ha)")
                st.markdown(f"<h2 style='color:green;font-size:30px;'>{agb_pred:.3f}</h2>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error computing bands/indices: {e}")