from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import streamlit as st
import matplotlib
# Thiết lập backend không interactive để tránh crash trên server
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

import osmnx as ox
from shapely.geometry import box, shape

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

# Import openlocationcode (cần cài đặt: pip install openlocationcode)
try:
    from openlocationcode import openlocationcode as olc
except ImportError:
    st.error("Thiếu thư viện `openlocationcode`. Vui lòng chạy: `pip install openlocationcode`")
    st.stop()

# =========================
# Streamlit + OSMnx settings
# =========================
st.set_page_config(page_title="KMU by PlusCode (OSMnx v2)", page_icon="🗺️", layout="wide")

# Cấu hình OSMnx 2.0
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.timeout = 180

# =========================
# Session State
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
# PlusCode Helpers
# =========================
@dataclass(frozen=True)
class PlusCell:
    pluscode: str
    north: float
    south: float
    east: float
    west: float

def _codearea_bounds(area) -> Tuple[float, float, float, float]:
    return float(area.latitudeHi), float(area.latitudeLo), float(area.longitudeHi), float(area.longitudeLo)

def pluscell_from_point(lat: float, lon: float, code_len: int) -> PlusCell:
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
        raise ValueError(f"Invalid bbox: N={n}, S={s}, E={e}, W={w}")

    origin_lat, origin_lon, cell_h, cell_w = _snap_grid_origin(s, w, code_len)
    
    # Tính số lượng hàng/cột dự kiến
    n_rows = int(math.ceil((n - origin_lat) / cell_h)) + 2
    n_cols = int(math.ceil((e - origin_lon) / cell_w)) + 2

    uniq: Dict[str, PlusCell] = {}
    truncated = False

    # Quét lưới
    for r in range(n_rows):
        lat0 = origin_lat + r * cell_h
        if lat0 > n + cell_h: break
        lat_center = lat0 + cell_h / 2

        for c in range(n_cols):
            lon0 = origin_lon + c * cell_w
            if lon0 > e + cell_w: break
            lon_center = lon0 + cell_w / 2

            # Tạo cell từ tâm để đảm bảo tính nhất quán
            cell = pluscell_from_point(lat_center, lon_center, code_len)

            # Kiểm tra xem cell có thực sự giao với bbox yêu cầu không
            if not (cell.east < w or cell.west > e or cell.north < s or cell.south > n):
                uniq[cell.pluscode] = cell
                if len(uniq) >= max_cells:
                    truncated = True
                    break
        if truncated: break

    return list(uniq.values()), truncated

def filter_cells_by_polygon(cells: List[PlusCell], poly_wgs84) -> List[PlusCell]:
    out = []
    for c in cells:
        rect = box(c.west, c.south, c.east, c.north)
        if poly_wgs84.intersects(rect):
            out.append(c)
    return out


# =========================
# Core Logic (No caching on Graph object to avoid Streamlit pickle errors)
# =========================
def download_graph_tile(pluscode: str, bbox_nsew: Tuple[float, float, float, float], network_type: str):
    n, s, e, w = bbox_nsew
    # OSMnx v2.0.0 graph_from_bbox tham số là bbox=(west, south, east, north)
    # Lưu ý: Một số phiên bản dev dùng (north, south, east, west), nhưng v2 chuẩn là (W, S, E, N) hoặc named args.
    # Để an toàn nhất, ta dùng named arguments:
    try:
        # Thử API v2 chuẩn
        G = ox.graph.graph_from_bbox(bbox=(w, s, e, n), network_type=network_type, retain_all=True, simplify=True)
        return G
    except Exception as e:
        # Fallback hoặc bắt lỗi cụ thể
        raise e

def compose_graphs(graphs: List[nx.MultiDiGraph]) -> Optional[nx.MultiDiGraph]:
    valid_graphs = [g for g in graphs if g is not None and len(g) > 0]
    if not valid_graphs:
        return None
    return nx.compose_all(valid_graphs)

def compute_kmu(G: nx.MultiDiGraph) -> Tuple[float, int, int, nx.MultiDiGraph]:
    # Project sang UTM để tính mét chính xác
    Gp = ox.projection.project_graph(G)
    stats = ox.stats.basic_stats(Gp)
    km = float(stats.get("street_length_total", 0.0) / 1000.0)
    return km, Gp.number_of_nodes(), Gp.number_of_edges(), Gp


# =========================
# UI Layout
# =========================
st.title("🗺️ KMU — PlusCode Grid (OSMnx v2 Fixed)")

mode = st.radio("Chế độ chọn vùng", ["Place", "BBox", "Draw"], horizontal=True)

cL, cR = st.columns([1.1, 1])

with cL:
    network_type = st.selectbox("Network Type", ["drive", "drive_service", "walk", "bike", "all"], index=0)
    code_len = st.selectbox("PlusCode Length", [4, 6, 8, 10], index=1, help="Độ dài càng lớn, ô càng nhỏ.")
    max_cells = st.slider("Max Cells limit", 20, 3000, 600, 20)
    delay_s = st.slider("Delay (giây)", 0.0, 3.0, 0.1, 0.1)
    
    st.divider()
    if st.button("🧹 Clear kết quả"):
        st.session_state["result"] = None
        st.session_state["last_error"] = None
        st.rerun()

with cR:
    st.subheader("Thông tin")
    st.write(f"OSMnx version: **{ox.__version__}**")
    st.caption("Dữ liệu đồ thị được tải từ OpenStreetMap thông qua Overpass API.")

# --- Inputs ---
poly_wgs = None
bbox_nsew: Optional[Tuple[float, float, float, float]] = None

if mode == "Place":
    place = st.text_input("Nhập tên địa điểm", value="District 1, Ho Chi Minh City")
    if st.button("🚀 Tính toán"):
        st.session_state["job_pending"] = True
        st.session_state["job_params"] = {
            "mode": "place", "place": place, "network_type": network_type, 
            "code_len": code_len, "max_cells": max_cells, "delay_s": delay_s
        }

elif mode == "BBox":
    c1, c2, c3, c4 = st.columns(4)
    north = c1.number_input("North", value=10.8, format="%.4f")
    south = c2.number_input("South", value=10.7, format="%.4f")
    east  = c3.number_input("East",  value=106.75, format="%.4f")
    west  = c4.number_input("West",  value=106.65, format="%.4f")
    
    if st.button("🚀 Tính toán"):
        st.session_state["job_pending"] = True
        st.session_state["job_params"] = {
            "mode": "bbox", "bbox": (north, south, east, west), "network_type": network_type,
            "code_len": code_len, "max_cells": max_cells, "delay_s": delay_s
        }

else: # Mode Draw
    st.markdown("### 🗺️ Vẽ vùng trên bản đồ")
    # Tạo bản đồ trung tâm
    m = folium.Map(location=[10.7769, 106.7009], zoom_start=12)
    Draw(
        export=False,
        draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False, "rectangle": True, "polygon": True},
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    ret = st_folium(m, height=400, use_container_width=True, returned_objects=["last_active_drawing"])
    
    if ret and ret.get("last_active_drawing"):
        st.session_state["draw_geom"] = ret["last_active_drawing"]["geometry"]

    if st.session_state["draw_geom"]:
        # Preview bounds
        tmp_shape = shape(st.session_state["draw_geom"])
        w, s, e, n = tmp_shape.bounds
        st.info(f"Đã chọn vùng: N={n:.4f}, S={s:.4f}, E={e:.4f}, W={w:.4f}")
        
        if st.button("🚀 Tính toán"):
            st.session_state["job_pending"] = True
            st.session_state["job_params"] = {
                "mode": "draw", "geom": st.session_state["draw_geom"], 
                "network_type": network_type, "code_len": code_len, 
                "max_cells": max_cells, "delay_s": delay_s
            }

# =========================
# Execution Pipeline
# =========================
def run_pipeline(params: dict) -> dict:
    mode_local = params["mode"]
    
    # 1. Xác định BBox tổng
    poly_local = None
    bbox_local = None # (n, s, e, w)

    if mode_local == "place":
        try:
            gdf = ox.geocoder.geocode_to_gdf(params["place"])
            poly_local = gdf.geometry.iloc[0]
            w, s, e, n = poly_local.bounds
            bbox_local = (n, s, e, w)
        except Exception as e:
            raise ValueError(f"Không tìm thấy địa điểm: {e}")
            
    elif mode_local == "bbox":
        bbox_local = params["bbox"]
        
    else: # draw
        if not params.get("geom"): raise ValueError("Chưa vẽ vùng nào!")
        poly_local = shape(params["geom"])
        w, s, e, n = poly_local.bounds
        bbox_local = (n, s, e, w)

    # 2. Tạo lưới PlusCode
    n, s, e, w = bbox_local
    cells, truncated = pluscode_grid_for_bbox(n, s, e, w, params["code_len"], params["max_cells"])
    
    # Lọc cell nếu có polygon (Place/Draw)
    if poly_local is not None:
        cells = filter_cells_by_polygon(cells, poly_local)
        
    if not cells:
        raise ValueError("Không tìm thấy cell nào trong vùng chọn (vùng quá nhỏ hoặc filter sai).")

    # 3. Tải Graph từng ô
    rows = []
    graphs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_cells = len(cells)
    
    for i, cell in enumerate(cells):
        status_text.text(f"Đang tải tile {i+1}/{total_cells}: {cell.pluscode}")
        progress_bar.progress((i + 1) / total_cells)
        
        try:
            # Download không qua st.cache_resource, để osmnx cache file
            G = download_graph_tile(cell.pluscode, (cell.north, cell.south, cell.east, cell.west), params["network_type"])
            
            if G is not None and len(G) > 0:
                km, nn, ne, _ = compute_kmu(G)
                rows.append({"pluscode": cell.pluscode, "km": km, "nodes": nn, "edges": ne, "status": "OK"})
                graphs.append(G)
            else:
                rows.append({"pluscode": cell.pluscode, "km": 0.0, "nodes": 0, "edges": 0, "status": "EMPTY"})
                
        except Exception as ex:
            # Ghi lỗi ngắn gọn
            err_msg = str(ex)
            rows.append({"pluscode": cell.pluscode, "km": 0, "nodes": 0, "edges": 0, "status": "ERR"})
            print(f"Error at {cell.pluscode}: {err_msg}")
        
        time.sleep(params["delay_s"])
        
    status_text.empty()
    progress_bar.empty()

    df = pd.DataFrame(rows).sort_values("pluscode")
    
    # 4. Gộp Graph
    if not graphs:
        return {"df": df, "total_km": 0, "total_nodes": 0, "total_edges": 0, "G_proj": None, "truncated": truncated}
        
    G_all = compose_graphs(graphs)
    total_km, total_nodes, total_edges, G_proj = compute_kmu(G_all)
    
    return {
        "df": df, "total_km": total_km, "total_nodes": total_nodes, 
        "total_edges": total_edges, "G_proj": G_proj, "truncated": truncated
    }

# Trigger Job
if st.session_state["job_pending"] and st.session_state["job_params"]:
    st.session_state["job_pending"] = False
    st.session_state["last_error"] = None
    
    with st.spinner("Đang xử lý dữ liệu OSM..."):
        try:
            res = run_pipeline(st.session_state["job_params"])
            st.session_state["result"] = res
        except Exception as e:
            st.session_state["last_error"] = str(e)
            st.error(f"Lỗi: {e}")

# =========================
# Result Display
# =========================
st.divider()
st.subheader("📌 Kết quả")

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])

res = st.session_state["result"]

if res:
    if res["truncated"]:
        st.warning(f"⚠️ Đã đạt giới hạn {max_cells} ô. Kết quả chỉ là một phần của vùng chọn.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng chiều dài đường (KM)", f"{res['total_km']:,.2f}")
    c2.metric("Tổng Nodes", f"{res['total_nodes']:,}")
    c3.metric("Tổng Edges", f"{res['total_edges']:,}")

    with st.expander("Xem chi tiết từng ô PlusCode", expanded=False):
        st.dataframe(res["df"], use_container_width=True)
        csv = res["df"].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Tải CSV chi tiết", csv, "pluscode_stats.csv", "text/csv")

    # Plotting
    if res["G_proj"]:
        st.write("### 🕸️ Bản đồ mạng lưới đường (Gộp)")
        # Sử dụng matplotlib figure một cách an toàn
        try:
            fig, ax = ox.plot.plot_graph(
                res["G_proj"], 
                show=False, 
                close=True, 
                node_size=0, 
                edge_linewidth=0.5, 
                edge_color="#F35B04", 
                bgcolor="#1E1E1E"
            )
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Không thể vẽ đồ thị: {e}")
