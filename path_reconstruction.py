from google.colab import files

# To upload a file
uploaded = files.upload()

from google.colab import drive
drive.mount('/content/drive')
destination_folder = "/content/drive/MyDrive/dataset/"
# import shutil
# import os

# for filename in uploaded.keys():
#     shutil.move(filename, os.path.join(destination_folder, filename))
#     print(f'File "{filename}" uploaded to "{destination_folder}" successfully.')

!pip install geopandas rasterio shapely numpy scikit-learn scikit-image matplotlib folium
! pip install contextily

!pip install --upgrade contextily xyzservices

import geopandas as gpd
import matplotlib.pyplot as plt
path = "/content/drive/MyDrive/dataset/"
temples = gpd.read_file(path + "temples.geojson")
roads = gpd.read_file(path + "roads.geojson")
forests = gpd.read_file(path + "Forests-landuse.geojson")
waterbodies = gpd.read_file(path + "WaterBodies.geojson")
import contextily as ctx
import matplotlib.pyplot as plt


temples = temples.to_crs(epsg=3857)
forests = forests.to_crs(epsg=3857)
waterbodies = waterbodies.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(12, 12))
forests.plot(ax=ax, color="#228B22", alpha=0.25)
waterbodies.plot(ax=ax, color="#1f78b4", alpha=0.4)
temples.plot(ax=ax, color="red", markersize=20, marker=".", zorder=3)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=11)
plt.show()



temples = temples.to_crs(epsg=3857)
forests = forests.to_crs(epsg=3857)
waterbodies = waterbodies.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))

forests.plot(ax=ax, color="#228B22", alpha=0.25, linewidth=0)
waterbodies.plot(ax=ax, color="#1f78b4", alpha=0.4, linewidth=0)
temples.plot(ax=ax, color="darkred", markersize=40, marker="o",
             edgecolor="white", linewidth=0.5, zorder=3)

xmin, ymin, xmax, ymax = temples.total_bounds
ax.set_xlim(xmin - 5000, xmax + 5000)
ax.set_ylim(ymin - 5000, ymax + 5000)

ctx.add_basemap(
    ax,
    source=ctx.providers.OpenStreetMap.Mapnik,
    zoom=12
)

ax.set_axis_off()
plt.tight_layout()
plt.show()
# Complete DEM Analysis for Temple Triangle
# Clean, production-ready code

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from rasterio.transform import rowcol
from shapely.geometry import Polygon, mapping
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your exact temple coordinates
TEMPLES = {
    'Thanjavur': (79.13855743408203, 10.786638259887695),
    'Kumbakonam': (79.38141632080078, 10.956926345825195),
    'Darasuram': (79.3443603515625, 10.948728561401367)
}

# File paths
DEM_PATH = "/content/drive/MyDrive/dataset/n10_e079_1arc_v3.tif"
OUTPUT_DIR = Path("/content/drive/MyDrive/dataset")
CLIPPED_DEM_PATH = OUTPUT_DIR / "triangle_dem_clipped.tif"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def hillshade(elevation, azimuth=315, angle_altitude=45):
    """Generate hillshade from elevation array."""
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(angle_altitude)

    x, y = np.gradient(elevation.astype('float64'))
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)

    shaded = (np.sin(alt) * np.sin(slope) +
              np.cos(alt) * np.cos(slope) * np.cos(az - aspect))

    return shaded

def get_elevation_robust(lon, lat, dem_array, transform, search_radius=3):
    """
    Extract elevation with nearby pixel fallback if exact point has NoData.
    """
    row, col = rowcol(transform, lon, lat)

    # Try exact pixel first
    if 0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1]:
        elev = dem_array[row, col]
        if elev != -32768 and not np.isnan(elev):
            return float(elev)

    # Search nearby pixels
    for r in range(max(0, row-search_radius), min(dem_array.shape[0], row+search_radius+1)):
        for c in range(max(0, col-search_radius), min(dem_array.shape[1], col+search_radius+1)):
            elev = dem_array[r, c]
            if elev != -32768 and not np.isnan(elev):
                return float(elev)

    return None

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*70)
print("COMPLETE DEM ANALYSIS - TEMPLE TRIANGLE")
print("="*70)

# Step 1: Load DEM and display basic info
print("\n[1] Loading DEM...")
with rasterio.open(DEM_PATH) as src:
    dem_full = src.read(1)
    src_crs = src.crs
    src_bounds = src.bounds
    src_res = src.res
    src_transform = src.transform

    print(f"    CRS: {src_crs}")
    print(f"    Bounds: {src_bounds}")
    print(f"    Resolution: {src_res}")
    print(f"    Shape: {dem_full.shape}")

# Step 2: Create triangle polygon and clip DEM
print("\n[2] Clipping DEM to triangle...")
triangle_coords = [TEMPLES['Thanjavur'], TEMPLES['Kumbakonam'], TEMPLES['Darasuram']]
triangle_poly = Polygon(triangle_coords)
triangle_geojson = [mapping(triangle_poly)]

with rasterio.open(DEM_PATH) as src:
    out_image, out_transform = mask(src, triangle_geojson, crop=True,
                                    nodata=-32768, filled=True)
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Save clipped DEM
    with rasterio.open(CLIPPED_DEM_PATH, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Clipped DEM saved: {CLIPPED_DEM_PATH}")

# Step 3: Load clipped DEM and calculate statistics
print("\n[3] Calculating terrain statistics...")
with rasterio.open(CLIPPED_DEM_PATH) as clipped:
    dem_data = clipped.read(1)
    dem_transform = clipped.transform
    dem_bounds = clipped.bounds

    # Create mask for NoData values
    dem_mask = (dem_data == -32768) | np.isnan(dem_data)
    valid_dem = dem_data[~dem_mask]

    print(f"    Minimum elevation: {valid_dem.min():.1f}m")
    print(f"    Maximum elevation: {valid_dem.max():.1f}m")
    print(f"    Mean elevation: {valid_dem.mean():.1f}m")
    print(f"    Median elevation: {np.median(valid_dem):.1f}m")
    print(f"    Relief (range): {valid_dem.max() - valid_dem.min():.1f}m")

# Step 4: Extract temple elevations
print("\n[4] Extracting temple elevations...")
for name, (lon, lat) in TEMPLES.items():
    elev = get_elevation_robust(lon, lat, dem_data, dem_transform)
    if elev:
        print(f"    {name}: {elev:.1f}m")
    else:
        print(f"    {name}: No valid data (check coordinates)")

# Step 5: Generate hillshade
print("\n[5] Generating hillshade...")
dem_for_hillshade = np.where(dem_mask, np.nan, dem_data)
hs = hillshade(dem_for_hillshade, azimuth=315, angle_altitude=45)
print("    Hillshade complete")

# Step 6: Create comprehensive visualization
print("\n[6] Creating visualizations...")

fig = plt.figure(figsize=(18, 12))

# Panel 1: Raw Elevation
ax1 = plt.subplot(2, 3, 1)
dem_img = ax1.imshow(dem_for_hillshade,
                     extent=[dem_bounds.left, dem_bounds.right,
                            dem_bounds.bottom, dem_bounds.top],
                     cmap='terrain', vmin=valid_dem.min(), vmax=valid_dem.max())

# Plot triangle
xs, ys = zip(*triangle_coords)
ax1.plot(list(xs) + [xs[0]], list(ys) + [ys[0]],
        'r--', linewidth=2.5, label='Triangle Boundary')

# Plot temples
for name, (lon, lat) in TEMPLES.items():
    ax1.scatter(lon, lat, marker='*', s=300, color='red',
               edgecolor='white', linewidth=2, zorder=5)
    ax1.annotate(name, (lon, lat), xytext=(5, 5),
                textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.set_title('Raw Elevation Data', fontsize=13, weight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend(loc='upper left', fontsize=8)
plt.colorbar(dem_img, ax=ax1, label='Elevation (m)', shrink=0.8)

# Panel 2: Hillshade
ax2 = plt.subplot(2, 3, 2)
ax2.imshow(hs, extent=[dem_bounds.left, dem_bounds.right,
                       dem_bounds.bottom, dem_bounds.top],
          cmap='gray')
ax2.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], 'r--', linewidth=2.5)
for lon, lat in triangle_coords:
    ax2.scatter(lon, lat, marker='*', s=300, color='red',
               edgecolor='white', linewidth=2, zorder=5)

ax2.set_title('Hillshade (Terrain Relief)', fontsize=13, weight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')

# Panel 3: Combined
ax3 = plt.subplot(2, 3, 3)
ax3.imshow(hs, extent=[dem_bounds.left, dem_bounds.right,
                       dem_bounds.bottom, dem_bounds.top],
          cmap='gray', alpha=0.6)
comb_img = ax3.imshow(dem_for_hillshade,
                      extent=[dem_bounds.left, dem_bounds.right,
                             dem_bounds.bottom, dem_bounds.top],
                      cmap='terrain', alpha=0.5)
ax3.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], 'r--', linewidth=2.5)
for lon, lat in triangle_coords:
    ax3.scatter(lon, lat, marker='*', s=300, color='yellow',
               edgecolor='black', linewidth=2, zorder=5)

ax3.set_title('Combined (Elevation + Hillshade)', fontsize=13, weight='bold')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')

# Panel 4: Elevation Histogram
ax4 = plt.subplot(2, 3, 4)
ax4.hist(valid_dem, bins=40, color='forestgreen', alpha=0.7, edgecolor='black')
ax4.axvline(valid_dem.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {valid_dem.mean():.1f}m')
ax4.axvline(np.median(valid_dem), color='blue', linestyle='--', linewidth=2,
           label=f'Median: {np.median(valid_dem):.1f}m')

ax4.set_xlabel('Elevation (m)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Elevation Distribution', fontsize=13, weight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# Panel 5: Slope Analysis
ax5 = plt.subplot(2, 3, 5)
dy, dx = np.gradient(dem_for_hillshade, src_res[0])
slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
slope_img = ax5.imshow(slope, extent=[dem_bounds.left, dem_bounds.right,
                                     dem_bounds.bottom, dem_bounds.top],
                      cmap='YlOrRd')
ax5.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], 'b--', linewidth=2)
ax5.set_title('Slope (degrees)', fontsize=13, weight='bold')
ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
plt.colorbar(slope_img, ax=ax5, label='Slope (°)', shrink=0.8)

# Panel 6: Statistics Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

stats_text = f"""
TERRAIN STATISTICS SUMMARY
{'='*35}

Elevation Statistics:
  Min:     {valid_dem.min():.1f} m
  Max:     {valid_dem.max():.1f} m
  Mean:    {valid_dem.mean():.1f} m
  Median:  {np.median(valid_dem):.1f} m
  Std Dev: {valid_dem.std():.1f} m
  Relief:  {valid_dem.max() - valid_dem.min():.1f} m

Slope Statistics:
  Mean:    {np.nanmean(slope):.2f}°
  Max:     {np.nanmax(slope):.2f}°

Terrain Classification:
  {np.nanmean(slope):.1f}° mean slope
  = {'Flat' if np.nanmean(slope) < 2 else 'Gently Rolling' if np.nanmean(slope) < 5 else 'Hilly'} terrain

Coverage:
  Valid pixels:  {(~dem_mask).sum():,}
  NoData pixels: {dem_mask.sum():,}
"""

ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Complete DEM Analysis - Thanjavur-Kumbakonam-Darasuram Triangle',
            fontsize=16, weight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])
output_fig = OUTPUT_DIR / "complete_dem_analysis.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"    Visualization saved: {output_fig}")
plt.show()

# Step 7: Summary
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nOutput files:")
print(f"  - Clipped DEM: {CLIPPED_DEM_PATH}")
print(f"  - Visualization: {output_fig}")
print(f"\nTerrain Summary: {valid_dem.max() - valid_dem.min():.1f}m relief, "
      f"{np.nanmean(slope):.1f}° average slope")
print("="*70)


# =========================
# Delta Region Full Pipeline
# =========================
# Paste this whole block into one cell in Colab / Jupyter and run.

# --- 0. Basic installs (uncomment if needed in Colab) ---
# !pip install --upgrade scikit-image ipywidgets geopandas rasterio contextily --quiet

# 1. IMPORTS
from pathlib import Path
import sys
from itertools import permutations
import math
import heapq

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.transform import rowcol
from rasterio.features import rasterize
from pyproj import Transformer, CRS
from shapely.geometry import Polygon, mapping, Point, LineString, box
from skimage.graph import MCP_Geometric
from IPython.display import display, HTML, clear_output

# ipywidgets for interactive UI
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout

import os
import subprocess
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

print("1) Running fc-cache (refresh system font cache)...")
try:
    # Refresh the system font cache (works on Linux/Colab)
    subprocess.run(["fc-cache", "-fv"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print("   fc-cache completed.")
except Exception as e:
    print("   fc-cache failed or not available on this system:", e)

# 2) Clear matplotlib font cache directory so it rebuilds (safe)
try:
    cache_dir = mpl.get_cachedir()
    fontlist_fname = os.path.join(cache_dir, "fontlist-v330.json")  # name may differ by mpl version
    # remove entire cache dir (matplotlib will recreate on demand)
    if os.path.isdir(cache_dir):
        print(f"2) Removing matplotlib cache dir: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
        except Exception as e:
            print("   Could not remove cache dir, trying to remove fontlist only:", e)
            try:
                if os.path.exists(fontlist_fname):
                    os.remove(fontlist_fname)
            except Exception as ee:
                print("   Also failed to remove fontlist:", ee)
    else:
        print("2) Matplotlib cache dir not present.")
except Exception as e:
    print("   Error clearing matplotlib cache:", e)

# 3) Set robust fallback fonts in rcParams (avoid Matplotlib hunting for missing fonts)
print("3) Setting robust fallback font configuration for Matplotlib...")
mpl.rcParams['font.family'] = 'sans-serif'
# Put known-good fonts first; DejaVu Sans is bundled with matplotlib and reliable.
mpl.rcParams['font.sans-serif'] = [
    'DejaVu Sans',        # default bundled fallback
    'Arial',              # common system font (if available)
    'Liberation Sans',
    'Noto Sans',
    'Sans'
]
# Ensure PDF/PS backends use same fallback
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42



# -------------------------
# 2. DEM utilities & analysis
# -------------------------

def hillshade(elevation, transform, azimuth=315, altitude=45):
    """
    Compute hillshade scaled to 0..1. Handles NaNs by local imputation (global mean).
    """
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    # pixel resolutions (positive)
    xres = abs(transform.a)
    yres = abs(transform.e)
    # fill NaNs with mean (safe for gradient calc)
    elev_f = np.where(np.isfinite(elevation), elevation, np.nanmean(elevation))
    dx, dy = np.gradient(elev_f, xres, yres)
    slope = np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)
    shaded = (np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return (shaded + 1.0) / 2.0


def safe_write_tiff(path, array, meta, nodata=-9999.0):
    """
    Write a single-band float32 GeoTIFF safely: replace NaN with nodata sentinel.
    """
    arr = array.copy().astype("float32")
    arr[np.isnan(arr)] = nodata
    meta2 = meta.copy()
    meta2.update({"count": 1, "dtype": "float32", "nodata": nodata})
    with rasterio.open(path, "w", **meta2) as dst:
        dst.write(arr, 1)


def run_dem_analysis(dem_path, temples_dict, output_dir):
    """
    Load DEM, clip to bounding box of temples (transformed to DEM CRS),
    produce clipped DEM, slope, hillshade, and return a results dict.
    """
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIPPED_DEM_PATH = OUTPUT_DIR / "triangle_dem_clipped.tif"

    print("=" * 60)
    print("🏔️ Starting DEM Analysis")
    print("=" * 60)
    try:
        with rasterio.open(dem_path) as src:
            src_crs = src.crs if src.crs is not None else CRS.from_epsg(4326)
            transformer = Transformer.from_crs(CRS.from_epsg(4326), src_crs, always_xy=True)

            # bounding box in DEM CRS using temple lon/lat
            coords_ll = list(temples_dict.values())
            xs = [c[0] for c in coords_ll]; ys = [c[1] for c in coords_ll]
            min_lon, max_lon = min(xs), max(xs)
            min_lat, max_lat = min(ys), max(ys)
            ul = transformer.transform(min_lon, max_lat)
            ur = transformer.transform(max_lon, max_lat)
            lr = transformer.transform(max_lon, min_lat)
            ll = transformer.transform(min_lon, min_lat)
            bounding_poly = Polygon([ul, ur, lr, ll])

            print("✂️  Clipping DEM to temple bounding box (projected to DEM CRS)...")
            out_image, out_transform = mask(src, [mapping(bounding_poly)], crop=True)
            # out_image: bands, rows, cols -> assume single-band DEM
            dem_arr = out_image[0].astype("float32")

            # handle DEM nodata
            src_nodata = src.nodata if src.nodata is not None else -32768
            dem_arr = np.where(dem_arr == src_nodata, np.nan, dem_arr)

            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": dem_arr.shape[0],
                "width": dem_arr.shape[1],
                "transform": out_transform,
                # keep CRS as src_crs; nodata handled on write
            })
            safe_write_tiff(CLIPPED_DEM_PATH, dem_arr, meta, nodata=-9999.0)
            print(f"💾 Clipped DEM saved: {CLIPPED_DEM_PATH}")

        # reopen clipped DEM for analysis
        with rasterio.open(CLIPPED_DEM_PATH) as clipped:
            dem_data = clipped.read(1).astype("float32")
            dem_data[dem_data == clipped.nodata] = np.nan
            dem_transform = clipped.transform
            dem_bounds = clipped.bounds
            # slope (degrees)
            dem_for_slope = np.where(np.isfinite(dem_data), dem_data, np.nanmean(dem_data))
            xres = abs(dem_transform.a); yres = abs(dem_transform.e)
            dy, dx = np.gradient(dem_for_slope, xres, yres)
            slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

            hs = hillshade(dem_data, dem_transform)
            print("✅ DEM analysis complete.")
            return {
                "dem_data": dem_data,
                "dem_transform": dem_transform,
                "dem_bounds": dem_bounds,
                "hillshade": hs,
                "slope": slope,
                "clipped_dem_path": CLIPPED_DEM_PATH,
                "dem_crs": clipped.crs
            }
    except Exception as e:
        print(f"❌ DEM analysis failed: {e}")
        return None

class DeltaPathModeler:
    def __init__(self, dem_data, dem_transform, dem_crs, slope_data, roads_gdf=None, forests_gdf=None, waterbodies_gdf=None):
        self.dem_data = dem_data
        self.dem_transform = dem_transform
        self.dem_crs = dem_crs
        self.slope = slope_data
        self.roads = roads_gdf
        self.forests = forests_gdf
        self.waterbodies = waterbodies_gdf
        self.cost_surface = None

    def create_cost_surface(self, weights):
        """
        Build multiplicative cost surface from slope, forests, and water.
        Roads are intentionally ignored for historical modeling.
        weights: dict with keys for moderate_slope, steep_slope, forest, water.
        """
        print("\n🗺️ Creating cost surface...")
        # slope cost
        slope_cost = np.ones_like(self.slope, dtype=np.float32)
        slope_cost = np.where(self.slope > 15, weights.get("steep_slope", 50.0),
                              np.where(self.slope > 5, weights.get("moderate_slope", 10.0), 1.0)).astype(np.float32)

        forest_cost = np.ones_like(slope_cost, dtype=np.float32)
        water_cost = np.ones_like(slope_cost, dtype=np.float32)

        # Roads intentionally excluded for historical landscape modelling
        road_cost = np.ones_like(slope_cost, dtype=np.float32)

        # rasterize forests
        if self.forests is not None and len(self.forests) > 0:
            try:
                forests_proj = self.forests.to_crs(self.dem_crs)
                forest_shapes = ((geom, 1) for geom in forests_proj.geometry if geom is not None and not geom.is_empty)
                forest_raster = rasterize(forest_shapes, out_shape=self.dem_data.shape, transform=self.dem_transform, fill=0, dtype="uint8")
                forest_cost = np.where(forest_raster == 1, weights.get("forest", 5.0), 1.0)
            except Exception as e:
                print(f"⚠️ Forest rasterization error: {e}")

        # rasterize water
        if self.waterbodies is not None and len(self.waterbodies) > 0:
            try:
                water_proj = self.waterbodies.to_crs(self.dem_crs)
                water_shapes = ((geom, 1) for geom in water_proj.geometry if geom is not None and not geom.is_empty)
                water_raster = rasterize(water_shapes, out_shape=self.dem_data.shape, transform=self.dem_transform, fill=0, dtype="uint8")
                water_cost = np.where(water_raster == 1, weights.get("water", 1000.0), 1.0)
            except Exception as e:
                print(f"⚠️ Water rasterization error: {e}")

        # Combine costs
        cost = slope_cost * forest_cost * water_cost * road_cost
        cost[np.isnan(self.dem_data)] = np.inf  # Make NoData areas impassable
        self.cost_surface = cost.astype(np.float32) + 1e-9 # Add small value to avoid zero costs
        print("✅ Cost surface created.")


    def _heuristic(self, a, b):
        """Euclidean distance heuristic for A*."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def find_least_cost_path_Astar(self, src_rc, dst_rc):
        """
        Finds the least cost path between two points using the A* algorithm.

        Args:
            src_rc (tuple): The starting (row, col) coordinates.
            dst_rc (tuple): The destination (row, col) coordinates.

        Returns:
            tuple: A tuple containing (path, cost).
                   'path' is a list of (row, col) tuples, or None if no path is found.
                   'cost' is the total path cost, or np.inf if no path is found.
        """
        rows, cols = self.cost_surface.shape
        start = (int(src_rc[0]), int(src_rc[1]))
        goal = (int(dst_rc[0]), int(dst_rc[1]))

        g_cost = np.full((rows, cols), np.inf)
        g_cost[start] = 0

        came_from = {}
        # The priority queue stores tuples of (f_score, node)
        open_list = [(self._heuristic(start, goal), start)]

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], g_cost[goal]  # Return reversed path and its cost

            (cx, cy) = current

            # Check all 8 possible neighbors
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),          (0, 1),
                           (1, -1),  (1, 0), (1, 1)]:

                neighbor = (cx + dx, cy + dy)
                nx, ny = neighbor

                if not (0 <= nx < rows and 0 <= ny < cols):
                    continue

                terrain_cost = self.cost_surface[nx, ny]
                if terrain_cost == np.inf:
                    continue  # Impassable terrain

                # Cost to move from current to neighbor
                # Diagonal moves cost sqrt(2) (~1.414) times more than straight moves.
                step_cost = terrain_cost * (1.414 if dx != 0 and dy != 0 else 1.0)

                tentative_g_score = g_cost[current] + step_cost

                if tentative_g_score < g_cost[neighbor]:
                    # This path to neighbor is better than any previous one. Record it!
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))

        return None, np.inf # No path found

# -------------------------
# 4. Visualization helpers + interactive UI
# -------------------------

def ensure_projected_crs(dem_crs):
    """Return a projected CRS (meters) based on DEM CRS; fallback to WebMercator."""
    try:
        crs_obj = CRS(dem_crs)
    except Exception:
        return CRS.from_epsg(3857)
    if crs_obj.is_projected:
        return crs_obj
    return CRS.from_epsg(3857)


def bbox_from_temples(temples_gdf, dem_results, expand_factor=1.5, min_buffer_m=1000):
    """Compute bounding box around temples (projected to DEM CRS) and expand it."""
    dem_crs = dem_results['dem_crs']
    temples_proj = temples_gdf.to_crs(dem_crs)
    minx, miny, maxx, maxy = temples_proj.total_bounds
    width = maxx - minx; height = maxy - miny
    cx = (minx + maxx) / 2.0; cy = (miny + maxy) / 2.0
    new_half_w = max(width * expand_factor / 2.0, min_buffer_m)
    new_half_h = max(height * expand_factor / 2.0, min_buffer_m)
    return (cx - new_half_w, cy - new_half_h, cx + new_half_w, cy + new_half_h)


def km_to_map_units(width_km, height_km, center_lonlat, dem_crs):
    """Convert width/height in km around center lon/lat into DEM CRS units (meters)."""
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    cx, cy = transformer.transform(center_lonlat[0], center_lonlat[1])
    half_w = (width_km * 1000.0) / 2.0
    half_h = (height_km * 1000.0) / 2.0
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def clamp_to_dem(bounds, dem_results):
    """Clamp bbox to clipped DEM bounds."""
    dem_bounds = dem_results['dem_bounds']
    minx, miny, maxx, maxy = bounds
    minx = max(minx, dem_bounds.left)
    miny = max(miny, dem_bounds.bottom)
    maxx = min(maxx, dem_bounds.right)
    maxy = min(maxy, dem_bounds.top)
    return (minx, miny, maxx, maxy)


def plot_region(dem_results, bbox=None, temples_gdf=None, roads_gdf=None, forests_gdf=None, water_gdf=None,
                show_temples=True, show_roads=True, show_forests=True, show_water=True,
                cmap_dem='gist_earth', vmin_pct=2, vmax_pct=98, figsize=(10,10), title=None):
    """
    Plot hillshade + DEM (contrast stretched) and optional vector layers clipped to bbox.
    bbox in DEM CRS: (minx, miny, maxx, maxy). If None -> entire clipped DEM.
    """
    dem = dem_results['dem_data']; hs = dem_results['hillshade']; slope = dem_results['slope']; dem_transform = dem_results['dem_transform']; dem_bounds = dem_results['dem_bounds']

    if bbox is None:
        extent = [dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top]
    else:
        bbox = clamp_to_dem(bbox, dem_results)
        extent = [bbox[0], bbox[2], bbox[1], bbox[3]]

    # compute sample window to determine vmin/vmax
    left, right, bottom, top = extent[0], extent[1], extent[2], extent[3]
    r1, c1 = rowcol(dem_transform, left, top)
    r2, c2 = rowcol(dem_transform, right, bottom)
    r1, r2 = sorted([int(r1), int(r2)])
    c1, c2 = sorted([int(c1), int(c2)])
    r1 = np.clip(r1, 0, dem.shape[0]-1); r2 = np.clip(r2, 0, dem.shape[0]-1)
    c1 = np.clip(c1, 0, dem.shape[1]-1); c2 = np.clip(c2, 0, dem.shape[1]-1)
    sample = dem[r1:r2+1, c1:c2+1]
    valid = sample[np.isfinite(sample)]
    if valid.size == 0:
        vmin, vmax = np.nanmin(dem), np.nanmax(dem)
    else:
        vmin = np.percentile(valid, vmin_pct); vmax = np.percentile(valid, vmax_pct)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(hs, cmap='gray', extent=[dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top], alpha=0.9)
    ax.imshow(dem, cmap=cmap_dem, extent=[dem_bounds.left, dem_bounds.right, dem_bounds.bottom, dem_bounds.top], vmin=vmin, vmax=vmax, alpha=0.6)

    # clip vectors and plot
    def safe_plot(gdf, z, **kwargs):
        if gdf is None or len(gdf)==0:
            return
        try:
            clip = gdf.to_crs(dem_results['dem_crs']).cx[extent[0]:extent[1], extent[2]:extent[3]]
            if not clip.empty:
                clip.plot(ax=ax, zorder=z, **kwargs)
        except Exception:
            pass

    if show_water:
        safe_plot(water_gdf, 1, color='#aaccff', alpha=0.6)
    if show_forests:
        safe_plot(forests_gdf, 2, color='#cce6b3', alpha=0.5)
    if show_roads:
        safe_plot(roads_gdf, 3, color='gray', linewidth=0.8)
    if show_temples:
        safe_plot(temples_gdf, 5, marker='*', color='red', markersize=120, edgecolor='black')

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(title if title else "Map region", fontsize=14, weight='bold')
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# -------------------------
# ADDED CODE: Global visualization settings + "Apply to all maps" button
# -------------------------

# Global settings store (will be used by plot_region & other plotting functions)
GLOBAL_VIZ_SETTINGS = {
    "buffer_factor": 1.5,
    "cmap": "gist_earth",
    "vmin_pct": 2,
    "vmax_pct": 98,
    "show_temples": True,
    "show_roads": True,
    "show_forests": True,
    "show_water": True
}

def set_global_viz_settings(**kwargs):
    """Update the global visualization settings dict with provided values."""
    GLOBAL_VIZ_SETTINGS.update({k: v for k, v in kwargs.items() if k in GLOBAL_VIZ_SETTINGS})
    return GLOBAL_VIZ_SETTINGS.copy()

def get_global_viz_settings():
    """Return a copy of the current global settings."""
    return GLOBAL_VIZ_SETTINGS.copy()

# Wrapper around plot_region that uses global settings when kwarg not provided
def plot_region_with_globals(dem_results, bbox=None, temples_gdf=None, roads_gdf=None, forests_gdf=None, water_gdf=None,
                             cmap_dem=None, vmin_pct=None, vmax_pct=None, show_temples=None, show_roads=None,
                             show_forests=None, show_water=None, title=None, figsize=(10,10)):
    s = get_global_viz_settings()
    # Provide defaults from global settings if arguments are None
    cmap_dem = cmap_dem if cmap_dem is not None else s["cmap"]
    vmin_pct = vmin_pct if vmin_pct is not None else s["vmin_pct"]
    vmax_pct = vmax_pct if vmax_pct is not None else s["vmax_pct"]
    show_temples = show_temples if show_temples is not None else s["show_temples"]
    show_roads = show_roads if show_roads is not None else s["show_roads"]
    show_forests = show_forests if show_forests is not None else s["show_forests"]
    show_water = show_water if show_water is not None else s["show_water"]

    # If no bbox provided, compute using buffer_factor from globals
    if bbox is None:
        bbox = bbox_from_temples(temples_gdf, dem_results, expand_factor=s["buffer_factor"])
    # Call the existing plot_region function with these resolved parameters
    plot_region(dem_results, bbox=bbox, temples_gdf=temples_gdf, roads_gdf=roads_gdf,
                forests_gdf=forests_gdf, water_gdf=water_gdf,
                show_temples=show_temples, show_roads=show_roads, show_forests=show_forests, show_water=show_water,
                cmap_dem=cmap_dem, vmin_pct=vmin_pct, vmax_pct=vmax_pct, figsize=figsize, title=title)

# Updated interactive UI additions: "Apply to all maps" button + description panel
def visualization_ui_with_apply_all(dem_results, temples_gdf, roads_gdf, forests_gdf, water_gdf):
    """
    Enhanced visualization UI:
    - Includes an 'Apply to all maps' button which writes the current widget values
      into GLOBAL_VIZ_SETTINGS. Once applied, any function that uses plot_region_with_globals
      will honor these values automatically.
    - Includes a short description box explaining how to use the controls.
    """
    dem_crs = dem_results['dem_crs']

    # Widgets (same as before, but we will push to global on demand)
    buffer_slider = widgets.FloatSlider(value=GLOBAL_VIZ_SETTINGS["buffer_factor"], min=1.0, max=6.0, step=0.1, description='Buffer x', continuous_update=False)
    cmap_dropdown = widgets.Dropdown(options=['gist_earth', 'terrain', 'viridis', 'cividis'], value=GLOBAL_VIZ_SETTINGS["cmap"], description='DEM cmap')
    vmin_slider = widgets.IntSlider(value=GLOBAL_VIZ_SETTINGS["vmin_pct"], min=0, max=40, step=1, description='vmin pct', continuous_update=False)
    vmax_slider = widgets.IntSlider(value=GLOBAL_VIZ_SETTINGS["vmax_pct"], min=60, max=100, step=1, description='vmax pct', continuous_update=False)

    show_temples = widgets.Checkbox(value=GLOBAL_VIZ_SETTINGS["show_temples"], description='Temples')
    show_roads = widgets.Checkbox(value=GLOBAL_VIZ_SETTINGS["show_roads"], description='Roads')
    show_forests = widgets.Checkbox(value=GLOBAL_VIZ_SETTINGS["show_forests"], description='Forests')
    show_water = widgets.Checkbox(value=GLOBAL_VIZ_SETTINGS["show_water"], description='Water')

    temple_names = temples_gdf['name'].tolist()
    center_select = widgets.Dropdown(options=['Temple: '+n for n in temple_names] + ['Manual lon/lat'], value='Temple: '+temple_names[0], description='Center')
    width_km = widgets.FloatSlider(value=10.0, min=0.5, max=200.0, step=0.5, description='Width (km)', continuous_update=False)
    height_km = widgets.FloatSlider(value=8.0, min=0.5, max=200.0, step=0.5, description='Height (km)', continuous_update=False)
    manual_lon = widgets.FloatText(value=temples_gdf.geometry.x.iloc[0], description='Lon')
    manual_lat = widgets.FloatText(value=temples_gdf.geometry.y.iloc[0], description='Lat')

    update_btn = widgets.Button(description='Update Map', button_style='primary')
    apply_all_btn = widgets.Button(description='Apply to all maps', button_style='success')
    section_btn = widgets.Button(description='Show Section', button_style='info')

    # Description / help text
    help_html = """
    <b>Visualization Controls — Quick Guide</b><br>
    - <i>Buffer x</i>: expand the temple bounding box (1.0 = tight box, higher = larger region).<br>
    - <i>DEM cmap</i>, <i>vmin/vmax pct</i>: contrast stretch using region percentiles.<br>
    - <i>Show ...</i>: toggle vector layers (temples, roads, forests, water).<br>
    - <i>Show Section</i>: display a rectangle centered on a selected temple (or manual lon/lat) with width/height in km.<br>
    - <b>Apply to all maps</b>: saves the current control values as the global defaults. Any subsequent call to <code>plot_region_with_globals()</code> or other UI-driven plots will use these settings automatically.<br>
    """

    help_box = widgets.HTML(value=help_html)
    out = widgets.Output(layout=Layout(border='1px solid lightgray'))

    def on_update(_=None):
        with out:
            clear_output(wait=True)
            factor = buffer_slider.value
            bbox = bbox_from_temples(temples_gdf, dem_results, expand_factor=factor)
            # Use the wrapper that respects global settings where appropriate
            plot_region_with_globals(dem_results, bbox=bbox, temples_gdf=temples_gdf, roads_gdf=roads_gdf,
                                     forests_gdf=forests_gdf, water_gdf=water_gdf,
                                     cmap_dem=cmap_dropdown.value, vmin_pct=vmin_slider.value, vmax_pct=vmax_slider.value,
                                     show_temples=show_temples.value, show_roads=show_roads.value,
                                     show_forests=show_forests.value, show_water=show_water.value,
                                     title=f"Buffered region (x{factor:.1f})")

    def on_section(_=None):
        with out:
            clear_output(wait=True)
            if center_select.value.startswith('Temple: '):
                name = center_select.value.replace('Temple: ', '')
                rec = temples_gdf[temples_gdf['name'] == name].iloc[0]
                center = (rec.geometry.centroid.x, rec.geometry.centroid.y)
            else:
                center = (manual_lon.value, manual_lat.value)
            bbox = km_to_map_units(width_km.value, height_km.value, center, dem_results['dem_crs'])
            bbox = clamp_to_dem(bbox, dem_results)
            plot_region_with_globals(dem_results, bbox=bbox, temples_gdf=temples_gdf, roads_gdf=roads_gdf,
                                     forests_gdf=forests_gdf, water_gdf=water_gdf,
                                     cmap_dem=cmap_dropdown.value, vmin_pct=vmin_slider.value, vmax_pct=vmax_slider.value,
                                     show_temples=show_temples.value, show_roads=show_roads.value,
                                     show_forests=show_forests.value, show_water=show_water.value,
                                     title=f"Section centered at {center}, {width_km.value}x{height_km.value} km")

    def on_apply_all(_=None):
        # Write current widget values into global settings
        new_settings = {
            "buffer_factor": buffer_slider.value,
            "cmap": cmap_dropdown.value,
            "vmin_pct": vmin_slider.value,
            "vmax_pct": vmax_slider.value,
            "show_temples": show_temples.value,
            "show_roads": show_roads.value,
            "show_forests": show_forests.value,
            "show_water": show_water.value
        }
        set_global_viz_settings(**new_settings)
        with out:
            clear_output(wait=True)
            print("✅ Global visualization settings updated:")
            for k, v in new_settings.items():
                print(f"   - {k}: {v}")
            print("\nTip: call plot_region_with_globals(...) in your other code to apply these settings automatically.")

    update_btn.on_click(on_update)
    section_btn.on_click(on_section)
    apply_all_btn.on_click(on_apply_all)

    # Layout + display
    row1 = HBox([buffer_slider, update_btn, apply_all_btn])
    row2 = HBox([cmap_dropdown, vmin_slider, vmax_slider])
    row3 = HBox([show_temples, show_roads, show_forests, show_water])
    row4 = HBox([center_select, width_km, height_km, section_btn])
    row5 = HBox([manual_lon, manual_lat])
    ui = VBox([help_box, row1, row2, row3, row4, row5, out])
    display(ui)

    # initial draw using current widget values
    on_update()

# -------------------------
# USAGE NOTE:
# - Call visualization_ui_with_apply_all(dem_results, temples_gdf, roads_gdf, forests_gdf, water_gdf)
# - Adjust controls; press "Update Map" to preview changes for the current view.
# - Press "Apply to all maps" to make the current widget values the global defaults.
# - Use plot_region_with_globals(...) elsewhere to render maps that automatically pick up global defaults.
# -------------------------


def plot_final_tour(dem_results, final_tour_path, tour_order, scenario, temples_gdf, roads_gdf, forests_gdf, waterbodies_gdf):
    """Creates a professional, clear, and focused plot of the final tour."""
    display(HTML(f"<h3>Step 5: Final Optimal Tour Visualization ({scenario})</h3>"))
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    # --- Auto-zoom to the tour area ---
    path_gdf = gpd.GeoDataFrame([{'geometry': final_tour_path}], crs=dem_results['dem_crs'])
    minx, miny, maxx, maxy = path_gdf.total_bounds
    if minx == maxx or miny == maxy:
        # fallback to DEM bounds if path bounding box degenerate
        bounds = dem_results['dem_bounds']
        minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top
    buffer = max((maxx - minx), (maxy - miny)) * 0.2 if (maxx - minx) > 0 and (maxy - miny) > 0 else 5000
    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)

    # --- Plotting Layers ---
    ax.imshow(dem_results['hillshade'], cmap='gray', alpha=0.7,
              extent=[dem_results['dem_bounds'].left, dem_results['dem_bounds'].right,
                      dem_results['dem_bounds'].bottom, dem_results['dem_bounds'].top])

    if waterbodies_gdf is not None and len(waterbodies_gdf) > 0:
        waterbodies_gdf.to_crs(dem_results['dem_crs']).plot(ax=ax, color='#aaccff', zorder=1)
    if forests_gdf is not None and len(forests_gdf) > 0:
        forests_gdf.to_crs(dem_results['dem_crs']).plot(ax=ax, color='#cce6b3', alpha=0.7, zorder=2)
    if roads_gdf is not None and len(roads_gdf) > 0:
        roads_gdf.to_crs(dem_results['dem_crs']).plot(ax=ax, color='black', linewidth=0.5, alpha=0.6, zorder=3)

    # --- ROUTE VISUALIZATION (fixed) ---
    # Draw the route using ax.plot from the LineString coordinates to avoid GeoPandas capstyle issues
    try:
        if isinstance(final_tour_path, LineString) and not final_tour_path.is_empty:
            xs, ys = final_tour_path.xy
            ax.plot(xs, ys, linewidth=7, color='black', solid_capstyle='round', zorder=4)
            ax.plot(xs, ys, linewidth=4, color='cyan', solid_capstyle='round', zorder=5)
            # add_arrows_to_line(ax, final_tour_path, arrow_size=(maxx-minx)/150) # This function was not defined, so it's commented out
        else:
            # fallback: try plotting via GeoDataFrame (without capstyle arg)
            path_gdf.plot(ax=ax, color='black', linewidth=7, zorder=4)
            path_gdf.plot(ax=ax, color='cyan', linewidth=4, zorder=5)
    except Exception as e:
        print("⚠️ Route plotting failed:", e)

    # --- Temple markers ---
    temples_proj = temples_gdf.to_crs(dem_results['dem_crs'])
    temples_proj.plot(ax=ax, marker='o', color='white', edgecolor='gray', markersize=30, zorder=6, alpha=0.7)
    for i, name in enumerate(tour_order):
        if name not in temples_proj['name'].values:
            continue
        temple_point = temples_proj[temples_proj['name'] == name].geometry.iloc[0]
        ax.plot(temple_point.x, temple_point.y, 'o', ms=30, mfc='yellow', mec='black', zorder=8)
        ax.text(temple_point.x, temple_point.y, f"{i+1}", fontsize=10, fontweight='bold', ha='center', va='center', color='black', zorder=9)

    # Legend and finishing touches
    legend_elements = [
        Line2D([0], [0], color='cyan', lw=4, label='Optimal Tour Route', solid_capstyle='round'),
        Line2D([0], [0], marker='o', color='yellow', label='Tour Stop', markeredgecolor='black', markersize=10, linestyle='None'),
        mpatches.Patch(color='#cce6b3', label='Forest'),
        mpatches.Patch(color='#aaccff', label='Water Body'),
        Line2D([0], [0], color='gray', lw=1, label='Road')
    ]
    ax.legend(handles=legend_elements, loc='upper left', prop={'size': 10, 'family': 'sans-serif'})
    title_str = ' → '.join(tour_order)
    ax.set_title(f"Optimal Tourist Tour ({scenario}):\n{title_str}", fontsize=20, weight='bold')
    plt.tight_layout()
    plt.show()


from ipyleaflet import Map, Marker, MarkerCluster, basemaps, basemap_to_tiles
from ipywidgets import VBox, HBox, SelectMultiple, Dropdown, Button, Output
from shapely.geometry import Point
import geopandas as gpd
import numpy as np

def ipyleaflet_tour_main(dem_results, temples_gdf, roads_gdf, forests_gdf, waterbodies_gdf, heuristic_weights):
    out = Output()

    temple_names = temples_gdf["name"].tolist()

    # Center map over temples
    center_coords = [temples_gdf.geometry.centroid.y.mean(), temples_gdf.geometry.centroid.x.mean()]
    m = Map(center=center_coords, zoom=12)
    m.add_layer(basemap_to_tiles(basemaps.OpenStreetMap.Mapnik))

    # Add temples as clustered markers
    cluster = MarkerCluster(name="Temples")
    for idx, row in temples_gdf.iterrows():
        marker = Marker(location=(row.geometry.centroid.y, row.geometry.centroid.x), title=row["name"])
        cluster.markers = cluster.markers + (marker,)
    m.add_layer(cluster)

    # Selection widgets
    temple_selector = SelectMultiple(
        options=[(name, idx) for idx, name in enumerate(temple_names)],
        description='Temples',
        rows=10
    )

    scenario_selector = Dropdown(
        options=list(heuristic_weights.keys()),
        value=list(heuristic_weights.keys())[0],
        description='Scenario'
    )

    run_button = Button(description="Compute Tour", button_style="success")

    control_panel = VBox([temple_selector, scenario_selector, run_button, out])
    ui = HBox([control_panel, m])

    display(ui)

    def run_tour(_):
        out.clear_output()
        with out:
            if len(temple_selector.value) < 2:
                print("❌ Select at least two temples!")
                return

            selected_indices = temple_selector.value
            tour_temples = [temple_names[i] for i in selected_indices]
            print(f"Selected temples: {tour_temples}")

            scenario_key = scenario_selector.value
            print(f"Scenario: {scenario_key}")

            modeler = DeltaPathModeler(
                dem_results["dem_data"], dem_results["dem_transform"], dem_results["dem_crs"],
                dem_results["slope"], roads_gdf, forests_gdf, waterbodies_gdf
            )
            modeler.create_cost_surface(heuristic_weights[scenario_key])

            # --- Compute all pairwise paths using A* ---
            pairwise_paths = {}
            print("Computing pairwise paths using A*...")
            temple_pairs = list(permutations(tour_temples, 2))

            tour_temples_gdf = temples_gdf[temples_gdf['name'].isin(tour_temples)].copy()
            tour_temples_proj = tour_temples_gdf.to_crs(dem_results['dem_crs'])

            for sname, ename in temple_pairs:
                if (sname, ename) in pairwise_paths: continue

                s_pt_proj = tour_temples_proj[tour_temples_proj['name'] == sname].geometry.iloc[0]
                e_pt_proj = tour_temples_proj[tour_temples_proj['name'] == ename].geometry.iloc[0]

                s_row, s_col = rowcol(dem_results['dem_transform'], s_pt_proj.x, s_pt_proj.y)
                e_row, e_col = rowcol(dem_results['dem_transform'], e_pt_proj.x, e_pt_proj.y)

                path_rc, cost = modeler.find_least_cost_path_Astar((s_row, s_col), (e_row, e_col))

                if path_rc:
                    path_xy = [dem_results['dem_transform'] * (c, r) for r, c in path_rc]
                    line = LineString(path_xy)
                    pairwise_paths[(sname, ename)] = (line, cost)
                    print(f"  Path found: {sname} -> {ename} (Cost: {cost:.1f})")
                else:
                    pairwise_paths[(sname, ename)] = (None, np.inf)
                    print(f"  ⚠️ No path found: {sname} -> {ename}")

            # --- Solve TSP with a greedy nearest-neighbor heuristic ---
            print("\nFinding best tour order (greedy nearest neighbor)...")
            start_node = tour_temples[0]
            unvisited = set(tour_temples[1:])
            tour_order = [start_node]
            total_cost = 0.0
            current = start_node

            while unvisited:
                next_node, min_cost = None, np.inf
                for candidate in unvisited:
                    path_info = pairwise_paths.get((current, candidate))
                    if path_info and path_info[1] < min_cost:
                        min_cost = path_info[1]
                        next_node = candidate

                if next_node is None:
                    print(f"❌ Could not find a path from {current} to any unvisited temple. Tour is incomplete.")
                    return

                tour_order.append(next_node)
                total_cost += min_cost
                unvisited.remove(next_node)
                current = next_node

            if len(tour_order) != len(tour_temples):
                print("❌ Could not compute a complete tour (some temples may be unreachable).")
                return

            # --- Assemble the final tour path for plotting ---
            final_coords = []
            for i in range(len(tour_order) - 1):
                seg, cost = pairwise_paths.get((tour_order[i], tour_order[i + 1]), (None, 0))
                if seg:
                    final_coords.extend(list(seg.coords))

            final_tour_path = LineString(final_coords)
            print(f"\n✅ Tour computed!")
            print(f"   Order: {' → '.join(tour_order)}")
            print(f"   Length: {final_tour_path.length/1000:.2f} km, Total Cost: {total_cost:.1f}")

            plot_final_tour(dem_results, final_tour_path, tour_order, scenario_key,
                            temples_gdf, roads_gdf, forests_gdf, waterbodies_gdf)

    run_button.on_click(run_tour)



def main():
    # IMPORTANT: Update this path to point to your data directory
    # For example, if using Google Drive: "/content/drive/MyDrive/my_geospatial_data/"
    data_path = "/content/drive/MyDrive/dataset"
    DEM_PATH = os.path.join(data_path, "n10_e079_1arc_v3.tif")
    OUTPUT_DIR = Path(data_path)

    # Check if files exist
    required_files = [
        DEM_PATH,
        os.path.join(data_path, "temples.geojson"),
        os.path.join(data_path, "roads.geojson"),
        os.path.join(data_path, "Forests-landuse.geojson"),
        os.path.join(data_path, "WaterBodies.geojson")
    ]
    if not all(os.path.exists(f) for f in required_files):
        print("❌ Error: One or more required data files are missing from the `data_path` directory.")
        print("Please ensure the following files are present:")
        for f in required_files:
            print(f" - {f}")
        return

    try:
        temples_gdf = gpd.read_file(os.path.join(data_path, "temples.geojson"))
        roads_gdf = gpd.read_file(os.path.join(data_path, "roads.geojson"))
        forests_gdf = gpd.read_file(os.path.join(data_path, "Forests-landuse.geojson"))
        waterbodies_gdf = gpd.read_file(os.path.join(data_path, "WaterBodies.geojson"))
    except Exception as e:
        print(f"❌ Failed to read vector files: {e}")
        return

    for gdf in [temples_gdf, roads_gdf, forests_gdf, waterbodies_gdf]:
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf.to_crs(epsg=4326, inplace=True)

    if "name" not in temples_gdf.columns:
        temples_gdf["name"] = [f"Temple_{i+1}" for i in range(len(temples_gdf))]
    else:
        temples_gdf["name"] = temples_gdf["name"].fillna("Unnamed Temple")

    temples_gdf = temples_gdf[temples_gdf.geometry.notna() & ~temples_gdf.geometry.is_empty].copy()
    if temples_gdf.empty:
        print("❌ No valid temple geometries found.")
        return

    TEMPLES = {
        row["name"]: (row.geometry.centroid.x, row.geometry.centroid.y)
        for _, row in temples_gdf.iterrows()
    }

    print(
        f"✅ Loaded {len(temples_gdf)} temples, "
        f"{len(roads_gdf)} roads, {len(forests_gdf)} forests, {len(waterbodies_gdf)} waterbodies."
    )

    dem_results = run_dem_analysis(DEM_PATH, TEMPLES, OUTPUT_DIR)
    if dem_results is None:
        print("❌ DEM processing failed; aborting.")
        return

    heuristic_weights = {
        "Historical (High-Cost Forest/Water)": {
            "moderate_slope": 10.0,
            "steep_slope": 50.0,
            "forest": 12.0,       # Higher cost to travel through forests
            "water": 5000.0       # Water is a major barrier
        },
        "Modern (Lower-Cost Forest)": {
            "moderate_slope": 6.0,
            "steep_slope": 20.0,
            "forest": 5.0,        # Assumes some paths/trails in forests
            "water": 1000.0       # Water is still a barrier, but perhaps more crossings exist
        }
    }

    ipyleaflet_tour_main(dem_results, temples_gdf, roads_gdf, forests_gdf, waterbodies_gdf, heuristic_weights)

if __name__ == "__main__":
    main()