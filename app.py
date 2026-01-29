# app.py
# Streamlit app: Road length (KMU) with OSMnx + PlusCode grid tiling
# A) PlusCode tiling with stable IDs
# B) Full app.py
# C) Interactive map to draw bbox/polygon + show pluscode grid
# D) Cache by pluscode for future re-queries

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box, Polygon, shape

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from openlocationcode import openlocationcode as olc


# =========================
# Streamlit / OSMnx settings
# =========================
st.set_page_config(page_title="KMU by PlusCode Grid (OSMnx)", page_icon="🗺️", layout="wide")

ox.settings.use_cache = True
ox.settings.log_console = False
try:
    ox.settings.overpass_rate_limit = True
except AttributeError:
    pass
ox.settings.timeout = 180

st.title("🗺️ Tính tổng chiều dài mạng lưới đường (KMU) — chia ô theo PlusCode")
st.caption("PlusCode (Open Location Code) tạo lưới toàn cầu có ID chuẩn; rất tiện lưu DB và truy vấn lại. "
           "Folium Draw + streamlit-folium cho phép vẽ vùng và trả bbox/geojson về Python. "
           "OSMnx dùng Overpass để tải mạng đường.")


# =========================
# Helpers / Compat (OSMnx 2.x)
# =========================
# OSMnx 2.x dùng namespaces; các API dưới đây theo v2.
graph_from_place = ox.graph.graph_from_place
graph_from_bbox_v2 = ox.graph.graph_from_bbox
basic_stats = ox.stats.basic_stats
plot_graph = ox.plot.plot_graph
geocode_to_gdf = ox.geocoder.geocode_to_gdf
project_graph = ox.projection.project_graph


# =========================
# PlusCode utilities
# =========================
@dataclass(frozen=True)
class PlusCell:
    pluscode: str
    north: float
    south: float
    east: float
    west: float

def pluscode_cell_from_point(lat: float, lon: float, code_len: int) -> PlusCell:
    """
    Encode (lat, lon) -> pluscode of length code_len, then decode -> bounding box.
    """
    code = olc.encode(lat, lon, code_len)
    area = olc.decode(code)  # gives latitudeLow/High, longitudeLow/High
    return PlusCell(
        pluscode=code,
        north=float(area.latitudeHigh),
        south=float(area.latitudeLow),
        east=float(area.longitudeHigh),
        west=float(area.longitudeLow),
    )

def snap_start_bbox_to_grid(north: float, south: float, east: float, west: float, code_len: int) -> Tuple[float, float]:
    """
    Snap start (south, west) down to the pluscode cell boundary so grid iteration aligns.
    """
    cell = pluscode_cell_from_point(south, west, code_len)
    return cell.south, cell.west

def iter_pluscode_grid_for_bbox(
    north: float, south: float, east: float, west: float,
    code_len: int,
    max_cells: int
) -> List[PlusCell]:
    """
    Generate pluscode cells that cover the bbox.
    We iterate aligned to the pluscode grid using decoded cell size.
    """
    # snap to grid boundary to avoid duplicates/misalignment
    start_lat, start_lon = snap_start_bbox_to_grid(north, south, east, west, code_len)

    # Determine cell size (degrees) by decoding one cell at start
    base = pluscode_cell_from_point(start_lat + 1e-9, start_lon + 1e-9, code_len)
    cell_h = max(1e-12, base.north - base.south)
    cell_w = max(1e-12, base.east - base.west)

    cells: List[PlusCell] = []
    lat = start_lat
    # Iterate rows
    while lat < north + cell_h:
        lon = start_lon
        while lon < east + cell_w:
            # Use cell center point to get stable code
            center_lat = lat + cell_h / 2
            center_lon = lon + cell_w / 2
            c = pluscode_cell_from_point(center_lat, center_lon, code_len)

            # Only keep if intersects requested bbox
            if not (c.east < west or c.west > east or c.north < south or c.south > north):
                cells.append(c)
                if len(cells) >= max_cells:
                    return cells

            lon += cell_w
        lat += cell_h

    # de-duplicate (can happen near edges due to float rounding)
    uniq = {}
    for c in cells:
        uniq[c.pluscode] = c
    return list(uniq.values())

def filter_cells_by_polygon(cells: List[PlusCell], poly_wgs84: Polygon) -> List[PlusCell]:
    """
    Remove cells that do not intersect the polygon (helps skip sea cells for islands).
    """
    out = []
    for c in cells:
        cell_poly = box(c.west, c.south, c.east, c.north)
        if poly_wgs84.intersects(cell_poly):
            out.append(c)
    return out


# =========================
# Download / Cache by pluscode (D)
# =========================
@st.cache_resource(show_spinner=False)
def download_graph_for_pluscode(pluscode: str, n: float, s: float, e: float, w: float, network_type: str):
    """
    Cached by (pluscode, bbox, network_type) => good for re-query later.
    Uses OSMnx v2 graph_from_bbox which expects bbox=(west,south,east,north).
    """
    G = graph_from_bbox_v2(
        bbox=(w, s, e, n),
        network_type=network_type,
        retain_all=True,
        simplify=True
    )
    return G

def compose_graphs(graphs: List[nx.MultiDiGraph]) -> Optional[nx.MultiDiGraph]:
    graphs = [g for g in graphs if g is not None and len(g) > 0]
    if not graphs:
        return None
    return nx.compose_all(graphs)

def compute_kmu(G: nx.MultiDiGraph) -> Tuple[float, int, int]:
    """
    Return (street_length_total_km, nodes, edges). Project first to avoid CRS warnings.
    """
    Gp = project_graph(G)
    s = basic_stats(Gp)
    km = float(s.get("street_length_total", 0.0) / 1000.0)
    return km, Gp.number_of_nodes(), Gp.number_of_edges()


# =========================
# UI: Mode selection
# =========================
mode = st.radio("Chế độ chọn vùng", ["Địa danh (Place)", "BBox nhập tay", "Vẽ vùng trên bản đồ"], horizontal=True)

left, right = st.columns([1.2, 1])

with left:
    network_type = st.selectbox(
        "network_type",
        ["drive", "drive_service", "all", "walk", "bike", "all_public"],
        index=1,
        help="Nếu tile rỗng nhiều, thử drive_service hoặc all."
    )

    code_len = st.selectbox(
        "Độ phân giải PlusCode (code length)",
        [4, 6, 8, 10],
        index=1,
        help=(
            "6: khoảng 0.05° (~vài km, tuỳ vĩ độ), 8: ~0.0025° (~trăm mét), "
            "10: ~0.000125° (~chục mét)."
        )
    )

    max_cells = st.slider("Giới hạn số ô PlusCode (max_cells)", 20, 2000, 400, 20)
    delay_s = st.slider("Delay giữa các request (Overpass friendly)", 0.0, 3.0, 0.6, 0.1)
    concurrency = st.slider("Song song tải tiles (khuyến nghị 1)", 1, 3, 1, 1)

    st.divider()
    st.subheader("Tùy chọn cache/log")
    use_cache = st.checkbox("OSMnx HTTP cache", value=True)
    debug_log = st.checkbox("OSMnx log_console", value=False)
    ox.settings.use_cache = use_cache
    ox.settings.log_console = debug_log

with right:
    st.subheader("Trạng thái")
    st.write(f"OSMnx: **{ox.__version__}**")
    st.write(f"PlusCode lib: **openlocationcode**")
    st.caption("Bạn có thể vẽ vùng để tránh lỗi geocode (Nominatim) khi bị chặn.")


# =========================
# Collect region geometry/bbox based on mode
# =========================
poly_wgs: Optional[Polygon] = None
bbox: Optional[Tuple[float, float, float, float]] = None  # (north, south, east, west)

if mode == "Địa danh (Place)":
    place = st.text_input("Nhập địa danh (ví dụ: Hanoi, Vietnam / Singapore)", value="Singapore")
    run = st.button("🚀 Tính KMU (Place)", type="primary")

    if run:
        with st.spinner("Geocoding địa danh (Nominatim)..."):
            # Geocode to polygon
            gdf = geocode_to_gdf(place)
            poly_wgs = gdf.geometry.iloc[0]
            west, south, east, north = poly_wgs.bounds
            bbox = (north, south, east, west)

elif mode == "BBox nhập tay":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        north = st.number_input("North (lat)", value=1.4700, format="%.6f")
    with c2:
        south = st.number_input("South (lat)", value=1.2000, format="%.6f")
    with c3:
        east = st.number_input("East (lon)", value=104.1000, format="%.6f")
    with c4:
        west = st.number_input("West (lon)", value=103.6000, format="%.6f")
    run = st.button("🚀 Tính KMU (BBox)", type="primary")
    if run:
        if north <= south or east <= west:
            st.error("BBox không hợp lệ: cần North>South và East>West.")
        else:
            bbox = (north, south, east, west)
            poly_wgs = None

else:
    st.markdown("### 🗺️ Vẽ vùng trên bản đồ (Rectangle/Polygon)")
    st.caption("Dùng Draw để vẽ rectangle/polygon. st_folium trả về bounds và last_active_drawing (GeoJSON).")

    # default center: Singapore
    center = [1.3521, 103.8198]
    m = folium.Map(location=center, zoom_start=11, control_scale=True, tiles="OpenStreetMap")

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": True,
            "polygon": True
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    map_ret = st_folium(m, height=520, use_container_width=True)

    # Extract drawing to polygon/bbox
    if map_ret and map_ret.get("last_active_drawing"):
        gj = map_ret["last_active_drawing"]
        geom = gj.get("geometry", None)
        if geom:
            poly_wgs = shape(geom)  # shapely geometry in WGS84
            west, south, east, north = poly_wgs.bounds
            bbox = (north, south, east, west)
            st.success(f"Đã nhận vùng vẽ. BBox: N={north:.5f}, S={south:.5f}, E={east:.5f}, W={west:.5f}")

    run = st.button("🚀 Tính KMU (Draw)", type="primary")


# =========================
# Main execution: PlusCode tiling + download + aggregate
# =========================
if run:
    if not bbox:
        st.warning("Chưa có vùng (bbox). Hãy nhập bbox hoặc vẽ vùng / geocode địa danh.")
        st.stop()

    north, south, east, west = bbox
    if north <= south or east <= west:
        st.error("BBox không hợp lệ.")
        st.stop()

    # 1) Generate pluscode grid cells for bbox
    with st.spinner("Đang tạo lưới PlusCode..."):
        cells = iter_pluscode_grid_for_bbox(north, south, east, west, code_len=code_len, max_cells=max_cells)

    # 2) If polygon exists (Place/Draw polygon), filter sea/outside cells
    if poly_wgs is not None:
        with st.spinner("Đang lọc ô theo polygon (bỏ ô biển/vùng ngoài)..."):
            cells = filter_cells_by_polygon(cells, poly_wgs)

    if not cells:
        st.error("Không tạo được ô PlusCode nào trong vùng.")
        st.stop()

    st.write(f"✅ Số ô PlusCode sẽ tải: **{len(cells)}** (code_len={code_len})")

    # =========================
    # C) Viewer: show grid on map (limited number to render)
    # =========================
    st.markdown("### 👀 Viewer: PlusCode grid (click để xem ID)")
    render_limit = min(len(cells), 400)  # prevent huge folium rendering
    m2 = folium.Map(location=[(north + south) / 2, (east + west) / 2], zoom_start=11, control_scale=True, tiles="OpenStreetMap")
    # Add bbox outline
    folium.Rectangle(bounds=[(south, west), (north, east)], color="#0000ff", weight=2, fill=False).add_to(m2)

    for c in cells[:render_limit]:
        folium.Rectangle(
            bounds=[(c.south, c.west), (c.north, c.east)],
            color="#ff0000",
            weight=1,
            fill=False,
            tooltip=c.pluscode,
        ).add_to(m2)

    st.caption(f"Hiển thị {render_limit}/{len(cells)} ô để tránh lag (tăng code_len hoặc giảm max_cells nếu muốn chi tiết hơn).")
    st_folium(m2, height=480, use_container_width=True)

    # =========================
    # Download tiles (D)
    # =========================
    st.markdown("### ⬇️ Tải dữ liệu đường theo PlusCode tiles")
    progress = st.progress(0.0)
    status = st.empty()

    results_rows: List[Dict] = []
    graphs: List[nx.MultiDiGraph] = []

    def fetch_one(idx: int, cell: PlusCell):
        try:
            G = download_graph_for_pluscode(cell.pluscode, cell.north, cell.south, cell.east, cell.west, network_type)
            if G is None or len(G) == 0:
                return {"pluscode": cell.pluscode, "km": 0.0, "nodes": 0, "edges": 0, "status": "EMPTY"}, None
            km, nn, ne = compute_kmu(G)
            return {"pluscode": cell.pluscode, "km": km, "nodes": nn, "edges": ne, "status": "OK"}, G
        except Exception as ex:
            return {"pluscode": cell.pluscode, "km": 0.0, "nodes": 0, "edges": 0, "status": f"ERR: {type(ex).__name__}: {ex}"}, None

    if concurrency == 1:
        for i, cell in enumerate(cells, start=1):
            status.text(f"Đang tải {i}/{len(cells)} • {cell.pluscode}")
            row, g = fetch_one(i, cell)
            results_rows.append(row)
            if g is not None:
                graphs.append(g)
            progress.progress(i / len(cells))
            time.sleep(delay_s)
    else:
        # limited concurrency, still with tiny delay per completion
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {pool.submit(fetch_one, i, cell): cell.pluscode for i, cell in enumerate(cells, start=1)}
            done = 0
            for fut in as_completed(futs):
                row, g = fut.result()
                results_rows.append(row)
                if g is not None:
                    graphs.append(g)
                done += 1
                progress.progress(done / len(cells))
                status.text(f"Đã xong {done}/{len(cells)}")
                time.sleep(delay_s)

    # Show per-cell report (with pluscode IDs)
    df = pd.DataFrame(results_rows).sort_values("pluscode")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Tải CSV theo PlusCode", df.to_csv(index=False).encode("utf-8"),
                       file_name="pluscode_tile_stats.csv", mime="text/csv")

    # =========================
    # Aggregate total
    # =========================
    st.markdown("### ✅ Tổng hợp KMU")
    if not graphs:
        st.error("Không tải được graph nào (tất cả tiles EMPTY/ERR). Thử tăng code_len (ô lớn hơn) hoặc đổi network_type.")
        st.stop()

    with st.spinner("Đang gộp graphs..."):
        G_all = compose_graphs(graphs)

    if G_all is None or len(G_all) == 0:
        st.error("Graph rỗng sau khi gộp.")
        st.stop()

    total_km, total_nodes, total_edges = compute_kmu(G_all)

    c1, c2, c3 = st.columns(3)
    c1.metric("🛣️ Tổng chiều dài (KMU)", f"{total_km:,.2f} km")
    c2.metric("Nodes", f"{total_nodes:,}")
    c3.metric("Edges", f"{total_edges:,}")

    st.markdown("### 🗺️ Plot graph (static)")
    fig, ax = plot_graph(project_graph(G_all), show=False, close=True, node_size=0, edge_linewidth=0.5, edge_color="#333", bgcolor="white")
    st.pyplot(fig)
