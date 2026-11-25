# Geographic data with Basemap
# Install: pip install basemap basemap-data-hires
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# City data: name, latitude, longitude
cities = {
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.08, 72.88),
    "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27),
    "Jodhpur": (26.24, 73.02),
}

# 1. World map
plt.figure(figsize=(10, 5))
m = Basemap(projection="cyl", resolution="c")
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color="lightgreen", lake_color="lightblue")
m.drawmapboundary(fill_color="lightblue")
plt.title("World map")
plt.show()

# 2. India map with cities
plt.figure(figsize=(7, 8))
m = Basemap(projection="merc", llcrnrlat=6, urcrnrlat=37,
            llcrnrlon=67, urcrnrlon=99, resolution="l")
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color="lightyellow", lake_color="lightblue")
m.drawmapboundary(fill_color="lightblue")

for name, (lat, lon) in cities.items():
    x, y = m(lon, lat)            # convert lat/lon to map coordinates
    m.plot(x, y, "ro")
    plt.text(x + 20000, y, name)

plt.title("Cities of India")
plt.show()
