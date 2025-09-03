# bands_indices.py
import rasterio
import numpy as np
from pyproj import Transformer

def compute_bands_and_indices(raster_path, lat, lon, width_m, height_m):
    """
    raster_path: path to tiff
    lat, lon: coordinates center
    width_m, height_m: plot size
    return: bands_dict, indices_dict
    """
    with rasterio.open(raster_path) as src:
        res_x, res_y = src.res
        img_crs = src.crs

        # แปลง lat/lon -> pixel
        transformer = Transformer.from_crs("EPSG:4326", img_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        col, row = ~src.transform * (x, y)
        col, row = int(col), int(row)

        # ขนาดหน้าต่างครอบพื้นที่
        width_px = max(1, int(width_m / res_x))
        height_px = max(1, int(height_m / abs(res_y)))
        half_w, half_h = width_px//2, height_px//2

        r0 = max(0, row - half_h)
        r1 = min(src.height, row + half_h + 1)
        c0 = max(0, col - half_w)
        c1 = min(src.width, col + half_w + 1)

        # อ่านเฉพาะ 4 bands: B,G,R,NIR
        band_indices = [1, 2, 3, 4]  # index เริ่มจาก 1 ใน rasterio
        bands_dict = {}
        for i, b in enumerate(["Blue (B1)", "Green (B2)", "Red (B3)", "NIR (B4)"]):
            band = src.read(band_indices[i])[r0:r1, c0:c1]
            mask = band != src.nodata if src.nodata is not None else np.ones_like(band, dtype=bool)
            mean_val = np.mean(band[mask])
            bands_dict[b] = float(mean_val)

        # ใช้ NIR = B4, R = B3
        NIR = bands_dict["NIR (B4)"]
        R = bands_dict["Red (B3)"]

        L = 0.5
        NDVI = (NIR - R) / (NIR + R + 1e-6)
        TNDVI = np.sqrt(NDVI + L)
        SR = NIR / (R + 1e-6)
        SAVI = ((1 + L)*(NIR - R)) / (NIR + R + L + 1e-6)
        MSAVI2 = (2*(NIR + 1) - np.sqrt((2*NIR + 1)**2 - 8*(NIR - R))) / 2

        indices_dict = {
            "NDVI": float(NDVI),
            "TNDVI": float(TNDVI),
            "SR": float(SR),
            "SAVI": float(SAVI),
            "MSAVI2": float(MSAVI2)
        }

        return bands_dict, indices_dict