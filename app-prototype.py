import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import joblib
from pyproj import Transformer

st.title("AGB Pixel Selection with Highlight")
# ---- Load ML model ----
model = joblib.load("svr_agb_model.pkl")
# ---- Upload raster ----
uploaded_file = st.file_uploader("Upload raster (.tif)", type="tif")
if uploaded_file:
   src = rasterio.open(uploaded_file)
   transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
   L = 0.5
   # Load RGB bands
   r = src.read(1).astype(np.float32)
   g = src.read(2).astype(np.float32)
   b = src.read(3).astype(np.float32)
   # Normalize each band 0-255
   def normalize_band(band):
       band_min, band_max = band.min(), band.max()
       return ((band - band_min) / (band_max - band_min + 1e-6) * 255).astype(np.uint8)
   rgb_display = np.stack([normalize_band(r), normalize_band(g), normalize_band(b)], axis=-1)
   # Display image
   st.subheader("Raster Image")
   fig, ax = plt.subplots()
   ax.imshow(rgb_display)
   ax.axis('off')
   st.pyplot(fig)
   # Select pixel via input
   st.subheader("Select Pixel (highlighted in red)")
   row = st.number_input("Row", min_value=0, max_value=src.height-1, value=src.height//2)
   col = st.number_input("Col", min_value=0, max_value=src.width-1, value=src.width//2)
   if st.button("Highlight & Compute"):
       # Draw rectangle
       fig, ax = plt.subplots()
       ax.imshow(rgb_display)
       rect = Rectangle((col-0.5, row-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
       ax.add_patch(rect)
       ax.axis('off')
       st.pyplot(fig)
       # Extract pixel values
       pixel = src.read()[:, row, col]  # [R,G,B,NIR]
       R,G,B,NIR = pixel
       R_ref, G_ref, B_ref, NIR_ref = [v/255.0 for v in [R,G,B,NIR]]
       # Compute indices
       ndvi   = (NIR_ref-R_ref)/(NIR_ref+R_ref+1e-6)
       tndvi  = np.sqrt((NIR_ref-R_ref)/(NIR_ref+R_ref+1e-6)+L)
       sr     = NIR_ref/(R_ref+1e-6)
       savi   = ((1+L)*(NIR_ref-R_ref))/(NIR_ref+R_ref+L+1e-6)
       msavi2 = (2*NIR_ref+1 - np.sqrt((2*NIR_ref+1)**2 - 8*(NIR_ref-R_ref)))/2
       # Predict AGB
       X = np.array([ndvi, tndvi, sr, savi, msavi2]).reshape(1,-1)
       agb_pred = float(model.predict(X)[0])
       # CRS coordinates
       x_crs, y_crs = src.xy(row, col)
       st.subheader("Pixel Information")
       st.write({
           "Row, Col": (row, col),
           "Pixel (R,G,B,NIR)": (R,G,B,NIR),
           "NDVI": ndvi,
           "TNDVI": tndvi,
           "SR": sr,
           "SAVI": savi,
           "MSAVI2": msavi2,
           "AGB (t/ha)": agb_pred,
           "CRS Coordinate": (x_crs, y_crs)
       })