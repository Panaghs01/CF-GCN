import rasterio
import rasterio.plot
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

A_path = "A/s2_040302.tif"
B_path = "B/s2_040302.tif"
mask_path = "MASKS/s2_040302_mask.tif"

a = rasterio.open(A_path)
b = rasterio.open(B_path)
mask = rasterio.open(mask_path)

rasterio.plot.show(a.read(),cmap='Greys')
rasterio.plot.show(b.read())
rasterio.plot.show(mask.read())

