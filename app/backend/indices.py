import numpy as np

def calculate_indices(red, nir, blue=None, green=None, L=0.5):
    """
    คำนวณ NDVI, TNDVI, SR, SAVI, MSAVI2
    """
    red = np.array(red, dtype=float)
    nir = np.array(nir, dtype=float)

    ndvi = (nir - red) / (nir + red + 1e-6)
    tndvi = np.sqrt(np.clip(ndvi + 0.5, a_min=0.0, a_max=None))
    sr = nir / (red + 1e-6)
    savi = ((1 + L) * (nir - red)) / (nir + red + L + 1e-6)
    msavi2 = (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2

    return {
        "NDVI": float(np.nanmean(ndvi)),
        "TNDVI": float(np.nanmean(tndvi)),
        "SR": float(np.nanmean(sr)),
        "SAVI": float(np.nanmean(savi)),
        "MSAVI2": float(np.nanmean(msavi2)),
    }