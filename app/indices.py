# indices.py
def calculate_indices(blue, green, red, nir):
    """คำนวณดัชนีพืชพรรณหลัก 5 ค่า (ให้รองรับค่าติดลบได้)"""
    try:
        ndvi = (nir - red) / (nir + red) if (nir + red) != 0 else 0
        tndvi = ((ndvi + 0.5) ** 0.5) if (ndvi + 0.5) >= 0 else -((-ndvi - 0.5) ** 0.5)
        sr = nir / red if red != 0 else 0
        savi = (1.5 * (nir - red)) / (nir + red + 0.5) if (nir + red + 0.5) != 0 else 0
        msavi2 = (2 * nir + 1 - ((2 * nir + 1) ** 2 - 8 * (nir - red)) ** 0.5) / 2 \
            if ((2 * nir + 1) ** 2 - 8 * (nir - red)) >= 0 else 0

    except Exception:
        ndvi, tndvi, sr, savi, msavi2 = 0, 0, 0, 0, 0

    return {
        "NDVI": ndvi,
        "TNDVI": tndvi,
        "SR": sr,
        "SAVI": savi,
        "MSAVI2": msavi2
    }