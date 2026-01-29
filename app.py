# app.py
# KMU (total street length) with OSMnx + PlusCode tiling + Map Draw viewer
# - A: PlusCode tiling (stable tile IDs)
# - B: Full app.py
# - C: Draw bbox/polygon on map + view pluscode grid
# - D: Cache by pluscode tile id
#
# Fixes:
# - openlocationcode CodeArea fields are latitudeLo/latitudeHi/longitudeLo/longitudeHi (NOT latitudeHigh/Low).  [1](https://github.com/google/open-location-code/blob/main/python/openlocationcode/openlocationcode.py)[2](https://deepwiki.com/google/open-location-code/3.5-python-implementation)
# - Add try/except around risky parts to prevent Streamlit Cloud auto-restart (“appear then disappear”).
# - st_folium returns dict with bounds and last_active_drawing for Draw plugin. [4](https://folium.streamlit.app/)[5](https://folium.streamlit.app/draw_support)
# - folium.plugins.Draw enables drawing rectangles/polygons. [3](https://python-visualization.github.io/folium/latest/user_guide/plugins/draw.html)[8](https://github.com/python-visualization/folium/blob/main/folium/plugins/draw.py)

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import streamlit as st
import osmnx as ox
from shapely.geometry import box, shape

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

from openlocationcode import openlocationcode as olc


# =========================
# Page & OSMnx settings
# =========================
st.set_page_config(page_title="KMU by PlusCode Grid (OSMnx)", page_icon="🗺️", layout="wide")

ox.settings.use_cache = True
ox.settings.log_console = False
try:
    ox.settings.overpass_rate_limit = True
except AttributeError:
    pass
ox.settings.timeout = 180

# OSMnx v2 namespaces
graph_from_place = ox.graph.graph_from_place
graph_from_bbox = ox.graph.graph_from_bbox
basic_stats = ox.stats.basic_stats
project_graph = ox.projection.project_graph
plot_graph = ox.plot.plot_graph
geocode_to_gdf = ox.geocoder.geocode_to_gdf


# =========================
# UI header
# =========================
st.title("🗺️ Tính tổng chiều dài mạng lưới đường (KMU) — PlusCode Grid")
st.caption(
    "Chia ô theo PlusCode (Open Location Code) để có tile ID chuẩn phục vụ lưu/truy vấn. "
    "Bạn có thể: Place / BBox / Vẽ vùng trên bản đồ."
)

# session state for draw geometry
if "draw_geom" not in st.session_state:
    st.session_state["draw_geom"] = None


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

def _codearea_bounds(area) -> Tuple[float, float, float, float]:
    """
    openlocationcode.decode returns CodeArea with attributes:
      latitudeLo, latitudeHi, longitudeLo, longitudeHi (Python implementation). [1](https://github.com/google/open-location-code/blob/main/python/openlocationcode/openlocationcode.py)[2](https://deepwiki.com/google/open-location-code/3.5-python-implementation)
    Return (north, south, east, west).
    """
    lat_lo = getattr(area, "latitudeLo", None)
    lat_hi = getattr(area, "latitudeHi", None)
    lon_lo = getattr(area, "longitudeLo", None)
    lon_hi = getattr(area, "longitudeHi", None)

    # Defensive fallback (should not be needed, but prevents crashes if environment differs)
    if lat_lo is None:
        lat_lo = getattr(area, "latitudeLow", getattr(area, "latitude_low", None))
    if lat_hi is None:
        lat_hi = getattr(area, "latitudeHigh", getattr(area, "latitude_high", None))
    if lon_lo is None:
        lon_lo = getattr(area, "longitudeLow", getattr(area, "longitude_low", None))
    if lon_hi is None:
        lon_hi = getattr(area, "longitudeHigh", getattr(area, "longitude_high", None))

    if None in (lat_lo, lat_hi, lon_lo, lon_hi):
        raise AttributeError(
            "Không đọc được CodeArea bounds. Cần latitudeLo/latitudeHi/longitudeLo/longitudeHi."
        )

    return float(lat_hi), float(lat_lo), float(lon_hi), float(lon_lo)

def pluscell_from_point(lat: float, lon: float, code_len: int) -> PlusCell:
    code = olc.encode(lat, lon, code_len)
    area = olc.decode(code)
    n, s, e, w = _codearea_bounds(area)
    return PlusCell(code, n, s, e, w)

def _snap_grid_origin(south: float, west: float, code_len: int) -> Tuple[float, float, float, float]:
    """
    Snap (south, west) down to the pluscode cell boundary. Also returns cell_h/cell_w in degrees.
    """
    base = pluscell_from_point(south + 1e-12, west + 1e-12, code_len)
    cell_h = max(1e-12, base.north - base.south)
    cell_w = max(1e-12, base.east - base.west)
    return base.south, base.west, cell_h, cell_w

def pluscode_grid_for_bbox(
    north: float, south: float, east: float, west: float,
    code_len: int, max_cells: int
) -> Tuple[List[PlusCell], bool]:
    """
    Generate unique pluscode cells covering bbox.
    Returns (cells, truncated_flag).
    """
    if north <= south or east <= west:
        raise ValueError("BBox không hợp lệ: cần North>South và East>West.")

    origin_lat, origin_lon, cell_h, cell_w = _snap_grid_origin(south, west, code_len)

    # number of steps (use ceil to avoid float drift)
    n_rows = int(math.ceil((north - origin_lat) / cell_h)) + 2
    n_cols = int(math.ceil((east - origin_lon) / cell_w)) + 2

    uniq: Dict[str, PlusCell] = {}
    truncated = False

    for r in range(n_rows):
        lat0 = origin_lat + r * cell_h
        lat_center = lat0 + cell_h / 2
        if lat0 > north + cell_h:
            break

        for c in range(n_cols):
            lon0 = origin_lon + c * cell_w
            lon_center = lon0 + cell_w / 2
            if lon0 > east + cell_w:
                break

            cell = pluscell_from_point(lat_center, lon_center, code_len)

            # bbox intersection check
            if not (cell.east < west or cell.west > east or cell.north < south or cell.south > north):
                uniq[cell.pluscode] = cell
                if len(uniq) >= max_cells:
                    truncated = True
                    break

        if truncated:
            break

    return list(uniq.values()), truncated

def filter_cells_by_polygon(cells: List[PlusCell], poly_wgs84) -> List[PlusCell]:
    """Skip sea/outside tiles: keep only cells intersecting polygon."""
    out = []
    for c in cells:
        rect = box(c.west, c.south, c.east, c.north)
        if poly_wgs84.intersects(rect):
            out.append(c)
    return out


# =========================
# Cache by pluscode tile (D)
# =========================
@st.cache_resource(show_spinner=False)
def download_graph_tile(pluscode: str, bbox_nsew: Tuple[float, float, float, float], network_type: str):
    """
    Cache key includes pluscode + bbox + network_type.
    OSMnx v2 graph_from_bbox expects bbox=(west,south,east,north). 
    """
    n, s, e, w = bbox_nsew
    G = graph_from_bbox(
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

def compute_kmu(G: nx.MultiDiGraph) -> Tuple[float, int, int, nx.MultiDiGraph]:
    """
    Project to planar CRS then compute street_length_total (km).
    """
    Gp = project_graph(G)
    stats = basic_stats(Gp)
    km = float(stats.get("street_length_total", 0.0) / 1000.0)
    return km, Gp.number_of_nodes(), Gp.number_of_edges(), Gp


# =========================
# Controls
# =========================
mode = st.radio("Chế độ chọn vùng", ["Địa danh (Place)", "BBox nhập tay", "Vẽ vùng trên bản đồ"], horizontal=True)

cL, cR = st.columns([1.1, 1])

with cL:
    network_type = st.selectbox(
        "network_type",
        ["drive_service", "drive", "all", "walk", "bike", "all_public"],
        index=0,
        help="Nếu tile rỗng nhiều, thử drive_service hoặc all."
    )
    code_len = st.selectbox(
        "Độ dài PlusCode (code_len)",
        [4, 6, 8, 10],
        index=1,
        help="6 vừa phải; 8 chi tiết; 10 rất chi tiết (tăng số tile rất nhanh)."
    )
    max_cells = st.slider("Giới hạn số tile (max_cells)", 20, 3000, 600, 20)
    delay_s = st.slider("Delay giữa request", 0.0, 3.0, 0.6, 0.1)
    concurrency = st.slider("Song song tải tiles", 1, 3, 1, 1)

    st.divider()
    use_cache = st.checkbox("OSMnx HTTP cache", value=True)
    debug_log = st.checkbox("OSMnx log_console", value=False)
    ox.settings.use_cache = use_cache
    ox.settings.log_console = debug_log

with cR:
    st.subheader("Trạng thái")
    st.write(f"OSMnx: **{ox.__version__}**")
    st.caption("Nếu geocode bị chặn (Nominatim), hãy dùng BBox hoặc Vẽ vùng.")


# =========================
# Region selection
# =========================
poly_wgs = None
bbox_nsew: Optional[Tuple[float, float, float, float]] = None  # (N,S,E,W)
run = False

if mode == "Địa danh (Place)":
    place = st.text_input("Nhập địa danh", value="Singapore")
    run = st.button("🚀 Tính (Place)", type="primary")

    if run:
        try:
            with st.spinner("Geocoding (Nominatim)..."):
                gdf = geocode_to_gdf(place)
            poly_wgs = gdf.geometry.iloc[0]
            west, south, east, north = poly_wgs.bounds
            bbox_nsew = (north, south, east, west)
        except Exception as ex:
            st.error("Geocode thất bại (có thể Nominatim bị chặn/timeout). Hãy dùng BBox hoặc Vẽ vùng.")
            st.exception(ex)
            st.stop()

elif mode == "BBox nhập tay":
    a, b, c, d = st.columns(4)
    north = a.number_input("North (lat)", value=1.4700, format="%.6f")
    south = b.number_input("South (lat)", value=1.2000, format="%.6f")
    east  = c.number_input("East (lon)", value=104.1000, format="%.6f")
    west  = d.number_input("West (lon)", value=103.6000, format="%.6f")

    run = st.button("🚀 Tính (BBox)", type="primary")
    if run:
        if north <= south or east <= west:
            st.error("BBox không hợp lệ: North>South và East>West.")
            st.stop()
        bbox_nsew = (north, south, east, west)
        poly_wgs = None

else:
    st.markdown("### 🗺️ Vẽ vùng (Rectangle/Polygon)")
    st.caption("Draw plugin cho phép vẽ shape. st_folium trả về last_active_drawing/bounds. [3](https://python-visualization.github.io/folium/latest/user_guide/plugins/draw.html)[4](https://folium.streamlit.app/)[5](https://folium.streamlit.app/draw_support)")

    center = [1.3521, 103.8198]
    m = folium.Map(location=center, zoom_start=11, control_scale=True, tiles="OpenStreetMap")

    Draw(
        export=False,
        draw_options={"polyline": False, "circle": False, "circlemarker": False, "marker": False, "rectangle": True, "polygon": True},
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    # keep reruns smaller by returning only needed objects (supported by streamlit-folium examples) [9](https://github.com/randyzwitch/streamlit-folium/blob/master/examples/pages/limit_data_return.py)
    ret = st_folium(m, height=520, use_container_width=True, returned_objects=["last_active_drawing", "bounds"])

    if ret and ret.get("last_active_drawing"):
        st.session_state["draw_geom"] = ret["last_active_drawing"].get("geometry")

    if st.session_state["draw_geom"]:
        try:
            poly_wgs = shape(st.session_state["draw_geom"])
            west, south, east, north = poly_wgs.bounds
            bbox_nsew = (north, south, east, west)
            st.success(f"BBox: N={north:.6f}, S={south:.6f}, E={east:.6f}, W={west:.6f}")
        except Exception as ex:
            st.error("Không đọc được polygon từ GeoJSON.")
            st.exception(ex)
            bbox_nsew = None
            poly_wgs = None

    run = st.button("🚀 Tính (Draw)", type="primary")


# =========================
# MAIN RUN
# =========================
if run:
    if not bbox_nsew:
        st.warning("Chưa có bbox. Hãy chọn vùng trước.")
        st.stop()

    north, south, east, west = bbox_nsew
    if north <= south or east <= west:
        st.error("BBox không hợp lệ.")
        st.stop()

    # 1) Create pluscode grid (error boundary => no auto-restart)
    try:
        with st.spinner("Đang tạo lưới PlusCode..."):
            cells, truncated = pluscode_grid_for_bbox(north, south, east, west, code_len, max_cells)
    except Exception as ex:
        st.error("Lỗi tạo PlusCode grid.")
        st.exception(ex)
        st.stop()

    # 2) Filter by polygon if available (skip sea/outside tiles)
    if poly_wgs is not None:
        try:
            with st.spinner("Đang lọc tiles theo polygon (bỏ ô biển/vùng ngoài)..."):
                cells = filter_cells_by_polygon(cells, poly_wgs)
        except Exception as ex:
            st.error("Lỗi lọc tiles theo polygon.")
            st.exception(ex)
            st.stop()

    if not cells:
        st.error("Không có tile nào trong vùng sau khi lọc.")
        st.stop()

    st.write(f"✅ Tiles: **{len(cells)}** (code_len={code_len})")
    if truncated:
        st.warning(f"⚠️ Đã chạm max_cells={max_cells}. Hãy thu hẹp vùng hoặc giảm độ chi tiết.")

    # 3) Viewer map for grid (limit to avoid lag)
    st.markdown("### 👀 Viewer: PlusCode Grid (hover để xem ID)")
    render_limit = min(len(cells), 400)
    m2 = folium.Map(location=[(north + south) / 2, (east + west) / 2], zoom_start=11, control_scale=True, tiles="OpenStreetMap")
    folium.Rectangle(bounds=[(south, west), (north, east)], color="#0000ff", weight=2, fill=False).add_to(m2)

    for c in cells[:render_limit]:
        folium.Rectangle(
            bounds=[(c.south, c.west), (c.north, c.east)],
            color="#ff0000",
            weight=1,
            fill=False,
            tooltip=c.pluscode,
        ).add_to(m2)

    st.caption(f"Hiển thị {render_limit}/{len(cells)} tiles để tránh lag.")
    st_folium(m2, height=480, use_container_width=True)

    # 4) Download tiles (D) + aggregate
    st.markdown("### ⬇️ Tải dữ liệu đường theo tiles")
    progress = st.progress(0.0)
    status = st.empty()

    rows: List[Dict] = []
    graphs: List[nx.MultiDiGraph] = []

    def fetch(cell: PlusCell):
        try:
            G = download_graph_tile(cell.pluscode, (cell.north, cell.south, cell.east, cell.west), network_type)
            if G is None or len(G) == 0:
                return {"pluscode": cell.pluscode, "km": 0.0, "nodes": 0, "edges": 0, "status": "EMPTY"}, None
            km, nn, ne, _ = compute_kmu(G)
            return {"pluscode": cell.pluscode, "km": km, "nodes": nn, "edges": ne, "status": "OK"}, G
        except Exception as ex:
            return {"pluscode": cell.pluscode, "km": 0.0, "nodes": 0, "edges": 0,
                    "status": f"ERR: {type(ex).__name__}: {ex}"}, None

    # sequential (Overpass friendly)
    if concurrency == 1:
        for i, cell in enumerate(cells, start=1):
            status.text(f"Đang tải {i}/{len(cells)} • {cell.pluscode}")
            row, g = fetch(cell)
            rows.append(row)
            if g is not None:
                graphs.append(g)
            progress.progress(i / len(cells))
            time.sleep(delay_s)
    else:
        # small concurrency (still delay per completion)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {pool.submit(fetch, cell): cell.pluscode for cell in cells}
            done = 0
            for fut in as_completed(futs):
                row, g = fut.result()
                rows.append(row)
                if g is not None:
                    graphs.append(g)
                done += 1
                progress.progress(done / len(cells))
                status.text(f"Đã xong {done}/{len(cells)}")
                time.sleep(delay_s)

    df = pd.DataFrame(rows).sort_values("pluscode")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Tải CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="pluscode_tile_stats.csv", mime="text/csv")

    ok_cnt = int((df["status"] == "OK").sum())
    empty_cnt = int((df["status"] == "EMPTY").sum())
    err_cnt = len(df) - ok_cnt - empty_cnt
    st.write(f"✅ OK: **{ok_cnt}** • ⬜ EMPTY: **{empty_cnt}** • ❌ ERR: **{err_cnt}**")

    if not graphs:
        st.error("Không tải được graph nào (tất cả tiles EMPTY/ERR). Thử giảm code_len (ô lớn hơn), đổi network_type, hoặc thu hẹp vùng.")
        st.stop()

    with st.spinner("Đang gộp graphs..."):
        G_all = compose_graphs(graphs)

    if G_all is None or len(G_all) == 0:
        st.error("Graph rỗng sau khi gộp.")
        st.stop()

    with st.spinner("Đang project & tính tổng..."):
        total_km, total_nodes, total_edges, G_proj = compute_kmu(G_all)

    c1, c2, c3 = st.columns(3)
    c1.metric("🛣️ Tổng chiều dài (KMU)", f"{total_km:,.2f} km")
    c2.metric("Nodes", f"{total_nodes:,}")
    c3.metric("Edges", f"{total_edges:,}")

    st.markdown("### 🗺️ Plot (static)")
    fig, ax = plot_graph(G_proj, show=False, close=True, node_size=0, edge_linewidth=0.5, edge_color="#333333", bgcolor="white")
    st.pyplot(fig)
