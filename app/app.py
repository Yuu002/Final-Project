# app.py
from flask import Flask, render_template, request, jsonify
import ee
import os
import joblib
import numpy as np
from indices import calculate_indices  # ใช้สูตรที่คุณเขียนไว้ใน indices.py

# -----------------------------
# Init Earth Engine
# -----------------------------
ee.Initialize(project='map-web-473508')

app = Flask(__name__)

# -----------------------------
# โหลดโมเดลและ Scaler
# -----------------------------
mlp_model = joblib.load(os.path.join("models","mlp_model.pkl"))
scaler_X = joblib.load(os.path.join("models","scaler_X.save"))
scaler_y = joblib.load(os.path.join("models","scaler_y.save"))

ref_model = joblib.load(os.path.join("models","ref_model.pkl"))
ref_scaler_X = joblib.load(os.path.join("models","ref_scaler_X.save"))
ref_scaler_y = joblib.load(os.path.join("models","ref_scaler_y.save"))

svr_model = joblib.load(os.path.join("models","svr_agb_model.pkl"))
svr_scaler_X = joblib.load(os.path.join("models","svr_scaler_X.save"))
svr_scaler_y = joblib.load(os.path.join("models","svr_scaler_y.save"))

# -----------------------------
# Cloud Mask Function (Sentinel-2 SR → Surface Reflectance 0–1)
# -----------------------------
def mask_s2_clouds_reflectance(image):
    """Masks clouds in Sentinel-2 SR and scales to reflectance (0–1)."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask).divide(10000)  # scale 0–1

# -----------------------------
# Cloud Mask + DN scale (Sentinel-2 → THEOS-style DN 0–100)
# -----------------------------
def mask_s2_clouds_dn(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    dn_image = image.divide(10000).multiply(100).toUint16()
    return dn_image.updateMask(mask)

# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -----------------------------
# Endpoint: ดึง Sentinel-2 และ THEOS-style map tiles
# -----------------------------
@app.route('/get_map', methods=['POST'])
def get_map():
    start = request.json['start']
    end = request.json['end']

    # Sentinel-2 Surface Reflectance (0–1)
    collection_ref = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_s2_clouds_reflectance)
    )
    image_ref = collection_ref.mean()

    vis_sentinel = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3, 'gamma': 1.0}
    map_id_sentinel = ee.Image(image_ref).getMapId(vis_sentinel)

    # THEOS-style DN
    collection_dn = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_s2_clouds_dn)
    )
    image_dn = collection_dn.mean()

    vis_theos = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 100, 'gamma': 1.0}
    map_id_theos = ee.Image(image_dn).getMapId(vis_theos)

    return jsonify({
        'sentinel_url': map_id_sentinel['tile_fetcher'].url_format,
        'theos_url': map_id_theos['tile_fetcher'].url_format
    })

# -----------------------------
# Endpoint: โหมด 1 จุด (Pixel)
# -----------------------------
@app.route('/get_pixel', methods=['POST'])
def get_pixel():
    lat = float(request.json['lat'])
    lon = float(request.json['lon'])
    start = request.json['start']
    end = request.json['end']

    point = ee.Geometry.Point([lon, lat])

    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_s2_clouds_reflectance)
    )

    image = collection.mean()

    bands = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=10
    )

    pixel_data = bands.getInfo()

    try:
        blue = float(pixel_data.get("B2", 0))
        green = float(pixel_data.get("B3", 0))
        red = float(pixel_data.get("B4", 0))
        nir = float(pixel_data.get("B8", 0))
    except:
        return jsonify({"error": "invalid pixel values"})

    # ---- Pipeline model ----
    X = np.array([[blue, green, red, nir]])
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = mlp_model.predict(X_scaled)
    field_reflectance = scaler_y.inverse_transform(y_pred_scaled)

    ref_X_scaled = ref_scaler_X.transform(field_reflectance)
    ref_y_pred_scaled = ref_model.predict(ref_X_scaled)
    ref_field_reflectance = ref_scaler_y.inverse_transform(ref_y_pred_scaled)
    ref_blue, ref_green, ref_red, ref_nir = ref_field_reflectance.ravel()

    indices = calculate_indices(ref_blue, ref_green, ref_red, ref_nir)

    idx_vals = np.array([[indices["NDVI"], indices["TNDVI"], indices["SR"], indices["SAVI"], indices["MSAVI2"]]])
    idx_scaled = svr_scaler_X.transform(idx_vals)
    agb_scaled = svr_model.predict(idx_scaled)
    agb = svr_scaler_y.inverse_transform(agb_scaled.reshape(-1, 1)).ravel()[0]

    pixel_data.update({
        "field_reflectance": {
            "Blue": ref_blue,
            "Green": ref_green,
            "Red": ref_red,
            "NIR": ref_nir
        },
        "indices": indices,
        "AGB": agb
    })

    return jsonify(pixel_data)

# -----------------------------
# Endpoint: โหมดพื้นที่ (Polygon) - รองรับทั้ง 'coords' และ GeoJSON 'geometry'
# -----------------------------
@app.route('/get_area', methods=['POST'])
def get_area():
    data = request.get_json(force=True)
    start = data.get('start')
    end = data.get('end')

    # อ่านพิกัดจากสองรูปแบบ: 'coords' (เดิม) หรือ 'geometry' (GeoJSON จาก leaflet.draw)
    coords = None
    if 'coords' in data:
        coords = data['coords']
    elif 'geometry' in data:
        geom = data['geometry']
        # GeoJSON polygon coordinates structure: [ [ [lon, lat], ... ] , ... ]
        coords = geom.get('coordinates')
    else:
        return jsonify({'error': 'No coords or geometry provided.'}), 400

    # สร้าง ee.Geometry.Polygon ให้ถูกต้องกับโครงสร้างที่ได้รับ
    try:
        # หาก coords เป็น list ของ points แบบ flat (เช่น [[lon,lat],...]) ให้ห่อเป็น ring เดียว
        if isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0][0], (int, float)):
            polygon = ee.Geometry.Polygon([coords])
        else:
            # กรณี coords มาเป็น list ของ rings หรือเป็น GeoJSON polygon coordinates
            polygon = ee.Geometry.Polygon(coords)
    except Exception as e:
        return jsonify({'error': f'Invalid geometry format: {str(e)}'}), 400

    # โหลด Sentinel-2 SR และกรองวันที่ + เมฆ
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mask_s2_clouds_reflectance)
    )
    image = collection.mean()

    # ให้ GEE เฉลี่ยค่าทุก pixel ใน polygon โดยตรง (ไม่จำกัดขนาด)
    try:
        result = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=polygon,
            scale=10,
            maxPixels=1e13,
            tileScale=4,
            bestEffort=True
        ).getInfo()
    except Exception as e:
        return jsonify({'error': f'GEE reduceRegion error: {str(e)}'}), 500

    blue = result.get('B2')
    green = result.get('B3')
    red = result.get('B4')
    nir = result.get('B8')

    if None in (blue, green, red, nir):
        return jsonify({'error': 'No valid pixel values found in this area.'}), 400

    # ---- Pipeline model ----
    X = np.array([[blue, green, red, nir]])
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = mlp_model.predict(X_scaled)
    field_reflectance = scaler_y.inverse_transform(y_pred_scaled)

    ref_X_scaled = ref_scaler_X.transform(field_reflectance)
    ref_y_pred_scaled = ref_model.predict(ref_X_scaled)
    ref_field_reflectance = ref_scaler_y.inverse_transform(ref_y_pred_scaled)
    ref_blue, ref_green, ref_red, ref_nir = ref_field_reflectance.ravel()

    indices = calculate_indices(ref_blue, ref_green, ref_red, ref_nir)

    idx_vals = np.array([[indices["NDVI"], indices["TNDVI"], indices["SR"], indices["SAVI"], indices["MSAVI2"]]])
    idx_scaled = svr_scaler_X.transform(idx_vals)
    agb_scaled = svr_model.predict(idx_scaled)
    agb = svr_scaler_y.inverse_transform(agb_scaled.reshape(-1, 1)).ravel()[0]

    # ขนาดพื้นที่
    try:
        area_sqm = polygon.area().getInfo()
    except Exception:
        area_sqm = None

    area_ha = (area_sqm / 10000) if area_sqm is not None else None
    area_rai = (area_sqm / 1600) if area_sqm is not None else None

    # ส่งผลลัพธ์ — ใช้คีย์เหมือน /get_pixel (AGB) เพื่อให้ frontend ไม่ต้องแก้
    return jsonify({
        "mean_reflectance": {
            "Blue": blue,
            "Green": green,
            "Red": red,
            "NIR": nir
        },
        "field_reflectance": {
            "Blue": ref_blue,
            "Green": ref_green,
            "Red": ref_red,
            "NIR": ref_nir
        },
        "indices": indices,
        "AGB": agb,
        "area": {
            "sqm": area_sqm,
            "hectare": area_ha,
            "rai": area_rai
        }
    })

# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)