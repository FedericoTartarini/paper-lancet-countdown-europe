import pprint

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio import features

from my_config import Dirs

# Set the path to your NUTS shapefile
shapefile = Dirs.shapefiles.value / "NUTS_RG_03M_2016_4326 clipped.geojson.zip"
nuts = gpd.read_file(shapefile)

# extract the NUTS2 regions
nuts2 = nuts[nuts.LEVL_CODE == 2].copy()

# Generate an integer ID from the NUTS2 code. In this case, the 4-character ID is encoded as a 32bit integer
nuts2["id_as_int"] = (
    nuts2.id.str.encode("utf-8")
    .apply(lambda b: np.frombuffer(b, dtype=">u4")[0])
    .astype(np.int32)
)

geom_with_id = [(row.geometry, row.id_as_int) for idx, row in nuts2.iterrows()]

# Define the top left (north west) point of the bounding box
north = 71.1
west = -31.2

# Define the pixel size
y_size = 0.1
x_size = 0.1
transform = rasterio.transform.from_origin(west, north, x_size, y_size)
# Define the output grid shape (using your reference climate grid for EU)
out_shape = (435, 761)

# Create a numpy array of the desired size
raster_nuts_grid = features.rasterize(
    geom_with_id, out_shape=out_shape, transform=transform, dtype=np.int32
)

# Generate some example data
input_array = np.random.random(out_shape)

# example for extracting data for a given region
nuts2_code = "UKM7"
row = nuts2[nuts2.id == nuts2_code]
id_as_int = row["id_as_int"].item()

# Plot the mask
mask = raster_nuts_grid == id_as_int

f, ax = plt.subplots(constrained_layout=True)
ax.imshow(mask)
plt.savefig(Dirs.figures_test.value / "nuts_grid.png")

# Plot the masked data
masked_array = input_array * mask
plt.imshow(masked_array)
plt.savefig(Dirs.figures_test.value / "masked_array.png")

# Example for summarising data for each nuts region
results = {}
for index, row in nuts2.iterrows():
    id_as_int = row["id_as_int"]
    mask = raster_nuts_grid == id_as_int

    # Apply a summary function, for example sum over the region.
    summary_for_masked_region = np.sum(mask * input_array)

    results[row["id"]] = summary_for_masked_region


# Use pprint to show a better formatting of the results
pprint.pprint(results)

nuts2.to_file(Dirs.rasters.value / "nuts2_with_integer_id.geojson", driver="GeoJSON")
np.savez(Dirs.rasters.value / "nuts2_integer_id_grid.npz", raster_nuts_grid)
