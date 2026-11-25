"""06 - Visualizing geographic data with Basemap

Install:
    pip install numpy pandas matplotlib basemap basemap-data basemap-data-hires
(Basemap is deprecated in favour of Cartopy but is still widely used in courses.)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# ---------------------------------------------------------------------
# Sample geographic data: major Indian cities (lat, lon, population in millions)
# ---------------------------------------------------------------------
cities = pd.DataFrame({
    "city": ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru",
             "Hyderabad", "Ahmedabad", "Jaipur", "Jodhpur", "Lucknow"],
    "lat": [28.61, 19.08, 22.57, 13.08, 12.97, 17.39, 23.02, 26.91, 26.24, 26.85],
    "lon": [77.21, 72.88, 88.36, 80.27, 77.59, 78.49, 72.57, 75.79, 73.02, 80.95],
    "pop": [32.9, 21.3, 15.1, 11.5, 13.2, 10.5, 8.0, 4.0, 1.5, 3.9],
})

# =====================================================================
# 1. World map - different projections
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (proj, title) in zip(axes, [("cyl", "Equidistant cylindrical"),
                                    ("robin", "Robinson"),
                                    ("moll", "Mollweide")]):
    m = Basemap(projection=proj, lon_0=0, resolution="c", ax=ax)
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.3)
    m.fillcontinents(color="lightgreen", lake_color="aqua")
    m.drawmapboundary(fill_color="aqua")
    m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0], fontsize=7)
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0, 0, 0, 1], fontsize=7)
    ax.set_title(title)
plt.tight_layout(); plt.show()

# =====================================================================
# 2. Orthographic (globe) view centred on India
# =====================================================================
plt.figure(figsize=(7, 7))
m = Basemap(projection="ortho", lat_0=22, lon_0=78, resolution="l")
m.drawcoastlines(); m.drawcountries()
m.drawmapboundary(fill_color="lightblue")
m.fillcontinents(color="wheat", lake_color="lightblue")
m.drawparallels(np.arange(-90, 91, 20)); m.drawmeridians(np.arange(-180, 181, 20))
x, y = m(cities["lon"].values, cities["lat"].values)
m.scatter(x, y, s=50, c="red", zorder=5)
plt.title("Globe view of India"); plt.show()

# =====================================================================
# 3. Regional map: India with cities (bubble size = population)
# =====================================================================
plt.figure(figsize=(9, 9))
m = Basemap(projection="merc", llcrnrlat=6, urcrnrlat=37,
            llcrnrlon=67, urcrnrlon=99, resolution="i")
m.drawmapboundary(fill_color="#cfe8ff")
m.fillcontinents(color="#f2efe6", lake_color="#cfe8ff")
m.drawcoastlines(linewidth=0.6)
m.drawcountries(linewidth=0.8)
m.drawstates(linewidth=0.3)
m.drawparallels(np.arange(5, 40, 5), labels=[1, 0, 0, 0], fontsize=8)
m.drawmeridians(np.arange(65, 101, 5), labels=[0, 0, 0, 1], fontsize=8)

x, y = m(cities["lon"].values, cities["lat"].values)
sc = m.scatter(x, y, s=cities["pop"] * 40, c=cities["pop"], cmap="YlOrRd",
               alpha=0.8, edgecolor="k", zorder=5)
plt.colorbar(sc, shrink=0.6, label="Population (millions)")
for name, xi, yi in zip(cities["city"], x, y):
    plt.text(xi + 15000, yi + 15000, name, fontsize=8)
plt.title("Major Indian cities"); plt.show()

# =====================================================================
# 4. Great-circle routes
# =====================================================================
plt.figure(figsize=(12, 6))
m = Basemap(projection="cyl", llcrnrlat=-60, urcrnrlat=80,
            llcrnrlon=-130, urcrnrlon=150, resolution="c")
m.bluemarble() if False else m.fillcontinents(color="lightgray", lake_color="white")
m.drawcoastlines(linewidth=0.4); m.drawcountries(linewidth=0.2)
m.drawmapboundary(fill_color="white")
routes = [((28.61, 77.21), (51.51, -0.13)),    # Delhi -> London
          ((28.61, 77.21), (40.71, -74.01)),   # Delhi -> New York
          ((28.61, 77.21), (35.68, 139.69))]   # Delhi -> Tokyo
for (la1, lo1), (la2, lo2) in routes:
    m.drawgreatcircle(lo1, la1, lo2, la2, linewidth=2, color="crimson")
    m.plot(*m([lo1, lo2], [la1, la2]), "bo", markersize=5)
plt.title("Great-circle routes from Delhi"); plt.show()

# =====================================================================
# 5. Gridded data: contour/filled-contour of a synthetic temperature field
# =====================================================================
lons = np.linspace(60, 100, 120)
lats = np.linspace(5, 40, 120)
lon_g, lat_g = np.meshgrid(lons, lats)
temp = 38 - 0.6 * (lat_g - 5) + 3 * np.sin(np.radians(lon_g * 4)) \
       + np.random.RandomState(1).randn(*lon_g.shape) * 0.3

plt.figure(figsize=(9, 8))
m = Basemap(projection="cyl", llcrnrlat=5, urcrnrlat=40,
            llcrnrlon=60, urcrnrlon=100, resolution="i")
cs = m.contourf(lon_g, lat_g, temp, levels=20, cmap="coolwarm")
m.drawcoastlines(); m.drawcountries()
m.drawparallels(np.arange(5, 41, 5), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(60, 101, 10), labels=[0, 0, 0, 1])
plt.colorbar(cs, label="Temperature (C) - synthetic")
plt.title("Filled contour on a map"); plt.show()

# =====================================================================
# 6. Relief / satellite backgrounds (need internet-free built-in images)
# =====================================================================
plt.figure(figsize=(10, 5))
m = Basemap(projection="mill", llcrnrlat=-60, urcrnrlat=85,
            llcrnrlon=-180, urcrnrlon=180, resolution="c")
m.shadedrelief()       # or m.bluemarble(), m.etopo()
m.drawcoastlines(linewidth=0.3)
plt.title("Shaded relief"); plt.show()

# =====================================================================
# 7. Day / night terminator
# =====================================================================
from datetime import datetime
plt.figure(figsize=(10, 5))
m = Basemap(projection="mill", lon_0=0)
m.drawcoastlines(); m.fillcontinents(color="tan", lake_color="skyblue")
m.drawmapboundary(fill_color="skyblue")
m.nightshade(datetime.utcnow(), alpha=0.35)
plt.title("Day/Night - " + datetime.utcnow().strftime("%d %b %Y %H:%M UTC"))
plt.show()

# =====================================================================
# 8. Reading a CSV of coordinates (your own data)
# =====================================================================
# df = pd.read_csv("my_locations.csv")        # needs columns lat, lon
# m = Basemap(projection="merc", llcrnrlat=df.lat.min()-2, urcrnrlat=df.lat.max()+2,
#             llcrnrlon=df.lon.min()-2, urcrnrlon=df.lon.max()+2, resolution="h")
# m.drawcoastlines(); m.scatter(*m(df.lon.values, df.lat.values), c="red")
# plt.show()
