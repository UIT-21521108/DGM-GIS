from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import box, shape

# =========================
# Streamlit + OSMnx settings
# =========================
st.set_page_config(page_title="KMU by PlusCode (OSMnx)", page_icon="🗺️", layout="wide")

ox.settings.use_cache = True
ox.settings.log_console = False
try:
    ox.settings.overpass_rate_limit = True
except AttributeError:
    pass
ox.settings.timeout = 180

# OSMnx v2 namespaces
graph_from_place = ox.graph.graph_from_place
graph_from_bbox  = ox.graph.graph_from_bbox
basic_stats      = ox.stats.basic_stats
project_graph    = ox.projection.project_graph
plot_graph       = ox.plot.plot_graph
geocode_to_gdf   = ox.geocoder.geocode_to_gdf


# =========================
# Session State (persist)
# =========================
if "draw_geom" not in st.session_state:
    st.session_state["draw_geom"] = None

if "job_pending" not in st.session_state:
    st.session_state["job_pending"] = False

if "job_params" not in st.session_state:
    st.session_state["job_params"] = None

if "result" not in st.session_state:
    st.session_state["result"] = None

if "last_error" not in st.session_state:
    st.session_state["last_error"] = None


# =========================
# Lazy imports (prevents crash at startup)
# =========================
def require(module_name: str, pip_name: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except Exception as ex:
        st.error(f"Thiếu hoặc lỗi import `{module_name}`. Hãy thêm `{pip_name}` vào requirements.txt.")
        st.code(str(ex))
        st.stop()

def get_olc():
    # openlocationcode.openlocationcode
    pkg = require("openlocationcode.openlocationcode", "openlocationcode")
    return pkg

def get_folium():
    folium = require("folium", "folium")
    plugins = require("folium.plugins", "folium")
    return folium, plugins

def get_streamlit_folium():
    sf = require("streamlit_folium", "streamlit-folium")
    return sf


# =========================
# PlusCode helpers
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
    openlocationcode.decode() CodeArea uses:
      latitudeLo/latitudeHi/longitudeLo/longitudeHi (Python impl). [3](https://github.com/google/open-location-code/blob/main/python/openlocationcode/openlocationcode.py)[4](https://deepwiki.com/google/open-location-code/3.5-python-implementation)
    Return (N,S,E,W)
    """
    lat_hi = float(area.latitudeHi)
    lat_lo = float(area.latitudeLo)
    lon_hi = float(area.longitudeHi)
    lon_lo = float(area.longitudeLo)
    return lat_hi, lat_lo, lon_hi, lon_lo

def pluscell_from_point(lat: float, lon: float, code_len: int) -> PlusCell:
    olc = get_olc()
    code = olc.encode(lat, lon, code_len)
    area = olc.decode(code)
    n, s, e, w = _codearea_bounds(area)
    return PlusCell(code, n, s, e, w)

def _snap_grid_origin(south: float, west: float, code_len: int) -> Tuple[float, float, float, float]:
    base = pluscell_from_point(south + 1e-12, west + 1e-12, code_len)
    cell_h = max(1e-12, base.north - base.south)
    cell_w = max(1e-12, base.east - base.west)
    return base.south, base.west, cell_h, cell_w

def pluscode_grid_for_bbox(n: float, s: float, e: float, w: float, code_len: int, max_cells: int) -> Tuple[List[PlusCell], bool]:
    if n <= s or e <= w:
        raise ValueError("BBox invalid: North>South & East>West required.")
    origin_lat, origin_lon, cell_h, cell_w = _snap_grid_origin(s, w, code_len)

    n_rows = int(math.ceil((n - origin_lat) / cell_h)) + 2
    n_cols = int(math.ceil((e - origin_lon) / cell_w)) + 2

    uniq: Dict[str, PlusCell] = {}
    truncated = False

    for r in range(n_rows):
        lat0 = origin_lat + r * cell_h
        if lat0 > n + cell_h:
            break
        lat_center = lat0 + cell_h / 2

        for c in range(n_cols):
            lon0 = origin_lon + c * cell_w
            if lon0 > e + cell_w:
                break
            lon_center = lon0 + cell_w / 2

            cell = pluscell_from_point(lat_center, lon_center, code_len)
            if not (cell.east < w or cell.west > e or cell.north < s or cell.south > n):
                uniq[cell.pluscode] = cell
                if len(uniq) >= max_cells:
                    truncated = True
                    break
        if truncated:
            break

    return list(uniq.values()), truncated

def filter_cells_by_polygon(cells: List[PlusCell], poly_wgs84) -> List[PlusCell]:
    out = []
    for c in cells:
        rect = box(c.west, c.south, c.east, c.north)
        if poly_wgs84.intersects(rect):
            out.append(c)
    return out


# =========================
# Cache per pluscode tile
# =========================
@st.cache_resource(show_spinner=False)
def download_graph_tile(pluscode: str, bbox_nsew: Tuple[float, float, float, float], network_type: str):
    n, s, e, w = bbox_nsew
    # OSMnx v2 expects bbox=(west, south, east, north)
    return graph_from_bbox(bbox=(w, s, e, n), network_type=network_type, retain_all=True, simplify=True)

def compose_graphs(graphs: List[nx.MultiDiGraph]) -> Optional[nx.MultiDiGraph]:
    graphs = [g for g in graphs if g is not None and len(g) > 0]
    if not graphs:
        return None
    return nx.compose_all(graphs)

def compute_kmu(G: nx.MultiDiGraph):
    Gp = project_graph(G)
    stats = basic_stats(Gp)
    km = float(stats.get("street_length_total", 0.0) / 1000.0)
    return km, Gp.number_of_nodes(), Gp.number_of_edges(), Gp


# =========================
# UI
# =========================
st.title("🗺️ KMU — PlusCode Grid (không biến mất khi rerun)")

mode = st.radio("Chế độ chọn vùng", ["Place", "BBox", "Draw"], horizontal=True)
colL, colR = st.columns([1.15, 1])

with colL:
    network_type = st.selectbox("network_type", ["drive_service", "drive", "all", "walk", "bike", "all_public"], index=0)
    code_len = st.selectbox("PlusCode code_len", [4, 6, 8, 10], index=1)
    max_cells = st.slider("max_cells", 20, 3000, 600, 20)
    delay_s = st.slider("delay_s", 0.0, 3.0, 0.6, 0.1)

    st.divider()
    if st.button("🧹 Clear kết quả"):
        st.session_state["result"] = None
        st.session_state["last_error"] = None
        st.session_state["job_pending"] = False
        st.session_state["job_params"] = None

with colR:
    st.subheader("Trạng thái")
    st.write(f"OSMnx: **{ox.__version__}**")
    st.caption("Kết quả lưu trong session_state nên rerun không mất. Nếu app vẫn restart, xem Logs để biết redeploy/crash.")


# =========================
# Input collection → set job_pending (NOT do heavy work here)
# =========================
if mode == "Place":
    place = st.text_input("Nhập địa danh", value="Singapore")
    if st.button("🚀 Tính"):
        st.session_state["job_pending"] = True
        st.session_state["job_params"] = {
            "mode": "place", "place": place,
            "network_type": network_type,
            "code_len": code_len,
            "max_cells": max_cells,
            "delay_s": delay_s,
        }

elif mode == "BBox":
    a, b, c, d = st.columns(4)
    north = a.number_input("North", value=1.4700, format="%.6f")
    south = b.number_input("South", value=1.2000, format="%.6f")
    east  = c.number_input("East",  value=104.1000, format="%.6f")
    west  = d.number_input("West",  value=103.6000, format="%.6f")

    if st.button("🚀 Tính"):
        st.session_state["job_pending"] = True
        st.session_state["job_params"] = {
            "mode": "bbox", "bbox": (north, south, east, west),
            "network_type": network_type,
            "code_len": code_len,
            "max_cells": max_cells,
            "delay_s": delay_s,
        }

else:
    st.markdown("### 🗺️ Vẽ vùng (Rectangle/Polygon)")
    folium, plugins = get_folium()
    sf = get_streamlit_folium()
    DrawPlugin = plugins.Draw

    m = folium.Map(location=[1.3521, 103.8198], zoom_start=11, control_scale=True, tiles="OpenStreetMap")
    DrawPlugin(
        export=False,
        draw_options={"polyline": False, "circle": False, "circlemarker": False, "marker": False, "rectangle": True, "polygon": True},
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    # Only return last_active_drawing to reduce reruns [2](https://github.com/randyzwitch/streamlit-folium/blob/master/examples/pages/limit_data_return.py)[1](https://folium.streamlit.app/)
    ret = sf.st_folium(m, height=520, use_container_width=True, returned_objects=["last_active_drawing"])

    if ret and ret.get("last_active_drawing"):
        st.session_state["draw_geom"] = ret["last_active_drawing"].get("geometry")

    if st.session_state["draw_geom"]:
        poly = shape(st.session_state["draw_geom"])
        w, s, e, n = poly.bounds
        st.info(f"BBox: N={n:.6f} S={s:.6f} E={e:.6f} W={w:.6f}")

    if st.button("🚀 Tính"):
        st.session_state["job_pending"] = True
        st.session_state["job_params"] = {
            "mode": "draw", "geom": st.session_state["draw_geom"],
            "network_type": network_type,
            "code_len": code_len,
            "max_cells": max_cells,
            "delay_s": delay_s,
        }


# =========================
# Pipeline (runs once, stores results)
# =========================
def run_pipeline(params: dict) -> dict:
    mode = params["mode"]
    network_type = params["network_type"]
    code_len = params["code_len"]
    max_cells = params["max_cells"]
    delay_s = params["delay_s"]

    poly_wgs = None
    bbox_nsew = None

    if mode == "place":
        gdf = geocode_to_gdf(params["place"])
        poly_wgs = gdf.geometry.iloc[0]
        w, s, e, n = poly_wgs.bounds
        bbox_nsew = (n, s, e, w)

    elif mode == "bbox":
        bbox_nsew = params["bbox"]

    else:
        geom = params.get("geom")
        if not geom:
            raise ValueError("Chưa có geometry từ Draw.")
        poly_wgs = shape(geom)
        w, s, e, n = poly_wgs.bounds
        bbox_nsew = (n, s, e, w)

    n, s, e, w = bbox_nsew
    cells, truncated = pluscode_grid_for_bbox(n, s, e, w, code_len, max_cells)
    if poly_wgs is not None:
        cells = filter_cells_by_polygon(cells, poly_wgs)
    if not cells:
        raise ValueError("Không có tile nào trong vùng.")

    rows = []
    graphs = []
    for i, cell in enumerate(cells, start=1):
        # status line
        row = {"pluscode": cell.pluscode, "km": 0.0, "nodes": 0, "edges": 0, "status": "EMPTY"}
        try:
            G = download_graph_tile(cell.pluscode, (cell.north, cell.south, cell.east, cell.west), network_type)
            if G is not None and len(G) > 0:
                km, nn, ne, _ = compute_kmu(G)
                row.update({"km": km, "nodes": nn, "edges": ne, "status": "OK"})
                graphs.append(G)
        except Exception as ex:
            row["status"] = f"ERR: {type(ex).__name__}: {ex}"
        rows.append(row)
        time.sleep(delay_s)

    df = pd.DataFrame(rows).sort_values("pluscode")
    G_all = compose_graphs(graphs)
    if G_all is None:
        raise ValueError("Tất cả tiles đều EMPTY/ERR.")

    total_km, total_nodes, total_edges, G_proj = compute_kmu(G_all)
    return {
        "truncated": truncated,
        "cells": len(cells),
        "df": df,
        "total_km": total_km,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "G_proj": G_proj,
    }


# Run once if pending
if st.session_state["job_pending"] and st.session_state["job_params"]:
    st.session_state["job_pending"] = False
    st.session_state["last_error"] = None
    try:
        with st.spinner("Đang tải dữ liệu & tính toán..."):
            st.session_state["result"] = run_pipeline(st.session_state["job_params"])
    except Exception as ex:
        st.session_state["result"] = None
        st.session_state["last_error"] = traceback.format_exc()


# =========================
# Always render result (never disappears on rerun)
# =========================
st.divider()
st.subheader("📌 Kết quả (persist)")

if st.session_state["last_error"]:
    st.error("Có lỗi xảy ra (xem traceback bên dưới):")
    st.code(st.session_state["last_error"])

res = st.session_state["result"]
if res:
    if res["truncated"]:
        st.warning("⚠️ Bị cắt bớt do max_cells. Thu hẹp vùng hoặc giảm độ chi tiết (code_len nhỏ hơn).")

    c1, c2, c3 = st.columns(3)
    c1.metric("🛣️ Tổng KMU", f"{res['total_km']:,.2f} km")
    c2.metric("Nodes", f"{res['total_nodes']:,}")
    c3.metric("Edges", f"{res['total_edges']:,}")

    st.dataframe(res["df"], use_container_width=True, hide_index=True)
    st.download_button("⬇️ CSV", res["df"].to_csv(index=False).encode("utf-8"), "pluscode_tile_stats.csv", "text/csv")

    fig, ax = plot_graph(res["G_proj"], show=False, close=True, node_size=0, edge_linewidth=0.5, edge_color="#333", bgcolor="white")
    st.pyplot(fig)
