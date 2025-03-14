import geopandas as gpd
import matplotlib.pyplot as plt

url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_03M_2021_4326_LEVL_3.geojson"
df = gpd.read_file(url)
f, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
df.plot(ax=ax)
plt.show()

url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_03M_2021_4326_LEVL_2.geojson"
df = gpd.read_file(url)
f, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
df.plot(ax=ax)
plt.show()
