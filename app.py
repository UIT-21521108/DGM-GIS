from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib

# Thiết lập backend không interactive để tránh crash trên Streamlit Cloud
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import osmnx as ox
from shapely.geometry import box, shape

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

# Import openlocationcode
try:
    from openlocationcode import openlocationcode as olc
except ImportError:
    st.error("Thiếu thư viện `openlocationcode`. Vui lòng chạy: `pip install openlocationcode`")
    st.stop()

# =========================
# Streamlit + OSMnx settings
# =========================
st.set_page_config(page_title="KMU PlusCode (Optimized)", page_icon="⚡", layout="wide")

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
    
    n_rows = int(math.ceil((n - origin_lat) / cell_h)) + 2
    n_cols = int(math.ceil((e - origin_lon) / cell_w)) + 2

    uniq: Dict[str, PlusCell] = {}
    truncated = False

    for r in range(n_rows):
        lat0 = origin_lat + r * cell_h
        if lat0 > n + cell_h: break
        lat_center = lat0 + cell_h / 2

        for c in range(n_cols):
            lon0 = origin_lon + c * cell_w
            if lon0 > e + cell_w: break
            lon_center = lon0 + cell_w / 2

            cell = pluscell_from_point(lat_center, lon_center, code_len)
            # Chỉ lấy các ô thực sự chạm vào bbox
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
# UI Layout
# =========================
st.title("⚡ KMU — PlusCode (Tối ưu hóa Batching)")
st.markdown("""
<style>
    div.stButton > button {width: 100%; background-color: #FF4B4B; color: white;}
</style>
""", unsafe_allow_html=True)

mode = st.radio("Chế độ chọn vùng", ["Place", "BBox", "Draw"], horizontal=True)

cL, cR = st.columns([1, 1.5])

with cL:
    network_type = st.selectbox("Loại đường (Network Type)", ["drive", "drive_service", "walk", "bike", "all"], index=0)
    code_len = st.selectbox("Độ dài PlusCode", [4, 6, 8, 10], index=1, help="6: Vừa phải (1.2km) | 8: Chi tiết (160m)")
    max_cells = st.number_input("Giới hạn số ô tối đa", value=2000, step=100)
    
    st.info("💡 **Mẹo:** Chế độ mới tải toàn bộ bản đồ 1 lần, không cần chỉnh Delay.")
    
    if st.button("🧹 Xóa kết quả"):
        st.session_state["result"] = None
        st.session_state["last_error"] = None
        st.rerun()

# --- Inputs ---
poly_wgs = None
bbox_nsew: Optional[Tuple[float, float, float, float]] = None

with cR:
    if mode == "Place":
        place = st.text_input("Nhập tên địa điểm", value="District 1, Ho Chi Minh City")
        if st.button("🚀 BẮT ĐẦU TÍNH TOÁN"):
            st.session_state["job_pending"] = True
            st.session_state["job_params"] = {
                "mode": "place", "place": place, "network_type": network_type, 
                "code_len": code_len, "max_cells": max_cells
            }

    elif mode == "BBox":
        c1, c2, c3, c4 = st.columns(4)
        north = c1.number_input("North", value=10.850, format="%.4f")
        south = c2.number_input("South", value=10.700, format="%.4f")
        east  = c3.number_input("East",  value=106.800, format="%.4f")
        west  = c4.number_input("West",  value=106.600, format="%.4f")
        
        if st.button("🚀 BẮT ĐẦU TÍNH TOÁN"):
            st.session_state["job_pending"] = True
            st.session_state["job_params"] = {
                "mode": "bbox", "bbox": (north, south, east, west), "network_type": network_type,
                "code_len": code_len, "max_cells": max_cells
            }

    else: # Mode Draw
        st.write("Vẽ hình chữ nhật hoặc đa giác lên bản đồ:")
        m = folium.Map(location=[10.7769, 106.7009], zoom_start=12)
        Draw(
            export=False,
            draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False, "rectangle": True, "polygon": True},
            edit_options={"edit": True, "remove": True},
        ).add_to(m)

        ret = st_folium(m, height=350, use_container_width=True, returned_objects=["last_active_drawing"])
        
        if ret and ret.get("last_active_drawing"):
            st.session_state["draw_geom"] = ret["last_active_drawing"]["geometry"]

        if st.session_state["draw_geom"]:
            # Preview bounds
            tmp_shape = shape(st.session_state["draw_geom"])
            w, s, e, n = tmp_shape.bounds
            st.success(f"Đã chọn vùng: N={n:.4f}, S={s:.4f}")
            
            if st.button("🚀 BẮT ĐẦU TÍNH TOÁN"):
                st.session_state["job_pending"] = True
                st.session_state["job_params"] = {
                    "mode": "draw", "geom": st.session_state["draw_geom"], 
                    "network_type": network_type, "code_len": code_len, 
                    "max_cells": max_cells
                }


# =========================
# CORE LOGIC: OPTIMIZED PIPELINE
# =========================
def run_pipeline_optimized(params: dict) -> dict:
    mode_local = params["mode"]
    network_type = params["network_type"]
    
    # --- BƯỚC 1: XÁC ĐỊNH BBOX TỔNG ---
    status = st.status("Đang xử lý dữ liệu...", expanded=True)
    
    poly_local = None
    bbox_local = None # (n, s, e, w)

    if mode_local == "place":
        status.write("📍 Đang tìm địa điểm (Geocoding)...")
        gdf_place = ox.geocoder.geocode_to_gdf(params["place"])
        poly_local = gdf_place.geometry.iloc[0]
        w, s, e, n = poly_local.bounds
        bbox_local = (n, s, e, w)
            
    elif mode_local == "bbox":
        bbox_local = params["bbox"]
        n, s, e, w = bbox_local
        
    else: # draw
        poly_local = shape(params["geom"])
        w, s, e, n = poly_local.bounds
        bbox_local = (n, s, e, w)

    # --- BƯỚC 2: TẠO LƯỚI GRID ---
    status.write("🕸️ Đang tạo lưới PlusCode...")
    n, s, e, w = bbox_local
    cells, truncated = pluscode_grid_for_bbox(n, s, e, w, params["code_len"], params["max_cells"])
    
    if poly_local is not None:
        cells = filter_cells_by_polygon(cells, poly_local)
        
    if not cells:
        status.update(label="❌ Lỗi: Không có ô lưới nào!", state="error")
        raise ValueError("Vùng chọn quá nhỏ hoặc không nằm trong phạm vi.")

    # --- BƯỚC 3: TẢI GRAPH TOÀN CỤC (1 LẦN DUY NHẤT) ---
    status.write(f"📥 Đang tải bản đồ từ OpenStreetMap ({len(cells)} ô lưới)... Vui lòng đợi.")
    
    # Download 1 lần cho toàn bộ bbox
    # OSMnx v2: bbox=(west, south, east, north)
    try:
        G_full = ox.graph.graph_from_bbox(bbox=(w, s, e, n), network_type=network_type, simplify=True)
    except Exception as ex:
        # Nếu không có dữ liệu (ví dụ chọn giữa biển), osmnx sẽ raise lỗi
        if "No data elements" in str(ex) or "found no graph nodes" in str(ex):
             status.update(label="⚠️ Vùng này không có đường!", state="complete")
             return {"df": pd.DataFrame(), "total_km": 0, "truncated": truncated, "G_proj": None}
        raise ex

    if len(G_full) == 0:
        status.update(label="⚠️ Bản đồ rỗng!", state="complete")
        return {"df": pd.DataFrame(), "total_km": 0, "truncated": truncated, "G_proj": None}

    # --- BƯỚC 4: PROJECT & CHUYỂN ĐỔI SANG GEODATAFRAME ---
    status.write("📐 Đang chuẩn hóa hệ tọa độ (UTM)...")
    G_proj = ox.projection.project_graph(G_full)
    
    # Lấy danh sách các cạnh (con đường)
    # nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)
    # Chúng ta chỉ quan tâm Edges để tính độ dài
    _, edges_gdf = ox.graph_to_gdfs(G_proj)
    
    # --- BƯỚC 5: CHUẨN BỊ GRID GEODATAFRAME ---
    status.write("✂️ Đang cắt bản đồ theo từng ô PlusCode...")
    
    # Tạo GeoDataFrame cho các ô lưới
    cell_data = []
    for c in cells:
        geom = box(c.west, c.south, c.east, c.north)
        cell_data.append({"pluscode": c.pluscode, "geometry": geom})
    
    gdf_cells = gpd.GeoDataFrame(cell_data, crs="EPSG:4326")
    # Chuyển hệ tọa độ của Grid sang trùng với Graph (UTM)
    gdf_cells_proj = gdf_cells.to_crs(edges_gdf.crs)

    # --- BƯỚC 6: CẮT HÌNH HỌC (OVERLAY INTERSECTION) ---
    # Kỹ thuật này cắt các con đường dài thành các đoạn nhỏ vừa khít với ô lưới
    # Giữ lại tính chính xác tuyệt đối
    
    try:
        # Overlay: Tìm phần giao nhau giữa Đường và Ô lưới
        # keep_geom_type=False để giữ cả LineString và MultiLineString
        intersections = gpd.overlay(edges_gdf, gdf_cells_proj, how='intersection', keep_geom_type=False)
        
        # Tính lại độ dài cho các đoạn vừa bị cắt (đơn vị mét -> km)
        intersections["segment_len_km"] = intersections.geometry.length / 1000.0
        
        # --- BƯỚC 7: TỔNG HỢP SỐ LIỆU ---
        stats = intersections.groupby("pluscode").agg(
            km=("segment_len_km", "sum"),
            count=("geometry", "count")
        ).reset_index()
        
        # Merge ngược lại với danh sách cell gốc để hiển thị cả những ô km=0
        df_final = pd.merge(gdf_cells[["pluscode"]], stats, on="pluscode", how="left").fillna(0)
        df_final = df_final.sort_values("pluscode")
        
    except Exception as e:
        # Fallback nếu overlay lỗi (hiếm gặp)
        status.write(f"⚠️ Lỗi cắt hình học: {e}. Đang dùng phương pháp thay thế...")
        df_final = pd.DataFrame(cell_data)
        df_final["km"] = 0
        df_final["count"] = 0

    total_km = df_final["km"].sum()
    total_edges = int(df_final["count"].sum())
    total_nodes = G_proj.number_of_nodes()

    status.update(label="✅ Hoàn tất!", state="complete")
    
    return {
        "df": df_final, 
        "total_km": total_km, 
        "total_nodes": total_nodes, 
        "total_edges": total_edges, 
        "G_proj": G_proj, 
        "truncated": truncated
    }

# =========================
# TRIGGER & DISPLAY
# =========================

if st.session_state["job_pending"] and st.session_state["job_params"]:
    st.session_state["job_pending"] = False
    st.session_state["last_error"] = None
    
    try:
        res = run_pipeline_optimized(st.session_state["job_params"])
        st.session_state["result"] = res
    except Exception as e:
        st.session_state["last_error"] = str(e)
        st.error(f"Đã xảy ra lỗi: {e}")

# --- Render Result ---
st.divider()

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])

res = st.session_state["result"]

if res:
    st.subheader("📊 Kết quả phân tích")
    
    if res.get("truncated"):
        st.warning(f"⚠️ Dữ liệu bị giới hạn {max_cells} ô. Hãy thu nhỏ vùng chọn để chính xác hơn.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng chiều dài", f"{res['total_km']:,.2f} km")
    col2.metric("Tổng đoạn đường", f"{res['total_edges']:,}")
    col3.metric("Tổng nút giao (Nodes)", f"{res['total_nodes']:,}")

    with st.expander("📂 Xem bảng dữ liệu chi tiết", expanded=True):
        st.dataframe(
            res["df"].style.format({"km": "{:.4f}", "count": "{:.0f}"}).background_gradient(subset=["km"], cmap="Greens"),
            use_container_width=True
        )
        csv = res["df"].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Tải file CSV", csv, "pluscode_kmu_stats.csv", "text/csv")

    # Vẽ biểu đồ
    if res["G_proj"]:
        st.write("### 🗺️ Bản đồ mạng lưới (Visualized)")
        with st.spinner("Đang vẽ bản đồ..."):
            try:
                fig, ax = ox.plot.plot_graph(
                    res["G_proj"], 
                    show=False, 
                    close=True, 
                    node_size=0, 
                    edge_linewidth=0.5, 
                    edge_color="#333", 
                    bgcolor="white"
                )
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Không thể hiển thị hình ảnh đồ thị lớn: {e}")
