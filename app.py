# app.py
# Streamlit app: Tính tổng chiều dài mạng lưới đường với OSMnx
# - Hỗ trợ: OSMnx v1.x và v2.x (Auto Detect)
# - Tính năng: Tối ưu vùng lớn bằng chia lưới (Tiling)
# - Fix lỗi: SyntaxError, Graph projection, Memory leak

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon

# Tắt warning không cần thiết của Geopandas/Shapely
warnings.filterwarnings("ignore")

# ============================================================
# 1) OSMnx COMPATIBILITY LAYER (Xử lý phiên bản)
# ============================================================
# Lấy phiên bản OSMnx để xử lý logic
try:
    OX_MAJOR_VERSION = int(ox.__version__.split(".")[0])
except:
    OX_MAJOR_VERSION = 1  # Mặc định về 1 nếu không parse được

def safe_graph_from_bbox(n, s, e, w, network_type):
    """
    Wrapper xử lý sự khác biệt giữa v1 và v2
    v1: graph_from_bbox(n, s, e, w, ...)
    v2: graph_from_bbox(bbox, ...)
    """
    if OX_MAJOR_VERSION >= 2:
        # OSMnx 2.x: Dùng tuple (north, south, east, west)
        return ox.graph.graph_from_bbox(bbox=(n, s, e, w), network_type=network_type)
    else:
        # OSMnx 1.x: Dùng 4 tham số rời
        try:
            return ox.graph_from_bbox(n, s, e, w, network_type=network_type)
        except AttributeError:
            # Fallback nếu import path khác
            return ox.graph.graph_from_bbox(n, s, e, w, network_type=network_type)

def safe_project_graph(G):
    """Wrapper cho hàm project_graph"""
    if OX_MAJOR_VERSION >= 2:
        return ox.project_graph(G)
    else:
        try:
            return ox.project_graph(G)
        except AttributeError:
            return ox.projection.project_graph(G)

def safe_basic_stats(G):
    """
    Wrapper cho basic_stats.
    v2.x đã bỏ tham số clean_int_tol.
    """
    if OX_MAJOR_VERSION >= 2:
        return ox.stats.basic_stats(G)
    else:
        # v1.x có thể dùng clean_int_tol, nhưng để an toàn ta bỏ qua
        return ox.basic_stats(G)

def safe_geocode(place_name):
    """Wrapper cho geocode"""
    if OX_MAJOR_VERSION >= 2:
        return ox.geocode_to_gdf(place_name)
    else:
        try:
            return ox.geocode_to_gdf(place_name)
        except AttributeError:
            return ox.geocoder.geocode_to_gdf(place_name)

# ============================================================
# 2) Streamlit UI
# ============================================================
st.set_page_config(
    page_title="Độ dài mạng lưới đường (OSMnx)",
    page_icon="🗺️",
    layout="centered",
)

# Cấu hình OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False
try:
    ox.settings.overpass_rate_limit = True 
except AttributeError:
    pass 
ox.settings.timeout = 180

st.title("🗺️ Tính chiều dài đường (OSMnx Auto-Compat)")
st.caption(f"Đang chạy OSMnx version: **{ox.__version__}**")

if "busy" not in st.session_state:
    st.session_state["busy"] = False
if "place_text" not in st.session_state:
    st.session_state["place_text"] = "Ho Chi Minh City, Vietnam"

mode = st.radio("Chọn chế độ nhập:", ["Địa danh (polygon)", "BBox"], horizontal=True)

colA, colB = st.columns([2, 1])
with colA:
    if mode == "Địa danh (polygon)":
        preset = st.selectbox(
            "Gợi ý mẫu",
            [
                "—",
                "District 1, Ho Chi Minh City, Vietnam",
                "Thu Duc City, Ho Chi Minh City, Vietnam",
                "Hue, Vietnam",
                "Hai Chau District, Danang, Vietnam",
                "Hanoi, Vietnam",
                "Singapore",
            ],
            index=0,
        )
        if preset != "—":
            st.session_state["place_text"] = preset
        place = st.text_input("Nhập tên địa danh:", key="place_text")
    else:
        place = ""

with colB:
    network_type = st.selectbox(
        "Loại đường",
        ["all", "all_public", "drive", "drive_service", "walk", "bike"],
        index=2, # Mặc định là drive
    )

with st.expander("⚙️ Tuỳ chọn nâng cao"):
    autosplit = st.checkbox("Tự chia nhỏ vùng lớn (Auto-Tiling)", True)
    area_threshold_km2 = st.number_input("Ngưỡng kích hoạt chia nhỏ (km²)", 1.0, 10000.0, 100.0, 10.0)
    tile_km = st.slider("Kích thước ô lưới (km)", 1, 25, 5, 1)
    max_tiles = st.slider("Giới hạn số ô tối đa", 4, 400, 100, 4)
    delay_s = st.slider("Delay giữa các request (s)", 0.0, 5.0, 0.5, 0.1)
    concurrency = st.slider("Số luồng tải song song (Thread)", 1, 5, 1, 1)
    show_tiles_outline = st.checkbox("Vẽ viền các ô lưới", True)

# ============================================================
# 3) HÀM XỬ LÝ LOGIC
# ============================================================
@st.cache_data(show_spinner=False)
def geocode_place_data(place_name: str):
    """Lấy dữ liệu địa lý của địa danh"""
    gdf = safe_geocode(place_name)
    # Project sang mét (Web Mercator) để tính diện tích
    gdf_webm = gdf.to_crs(3857)
    area_km2 = float(gdf_webm.area.iloc[0] / 1e6)
    return gdf, gdf_webm, area_km2

def poly_to_tiles(poly_m: Polygon, tile_km, max_tiles):
    """Chia Polygon (đơn vị mét) thành các ô lưới"""
    minx, miny, maxx, maxy = poly_m.bounds
    step = tile_km * 1000 # Đổi km sang mét

    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)

    bboxes = []
    # Loop qua lưới
    for x in xs:
        for y in ys:
            # Tạo ô vuông
            cell = box(x, y, x + step, y + step)
            # Kiểm tra giao cắt với vùng địa danh gốc
            if not poly_m.intersects(cell):
                continue
            
            # Lấy phần giao nhau
            inter = poly_m.intersection(cell)
            if inter.is_empty:
                continue
            
            # Chuyển ngược về WGS84 (Lat/Lon) để lấy BBox tải dữ liệu
            inter_wgs = gpd.GeoSeries([inter], crs=3857).to_crs(4326).iloc[0]
            lon_min, lat_min, lon_max, lat_max = inter_wgs.bounds
            
            # Lưu thứ tự: North, South, East, West
            bboxes.append((lat_max, lat_min, lon_max, lon_min))
            
            if len(bboxes) >= max_tiles:
                return bboxes
    return bboxes

def bbox_to_tiles(n, s, e, w, tile_km, max_tiles):
    """Chia BBox thành các tile nhỏ hơn"""
    poly = box(w, s, e, n) # shapely box: minx, miny, maxx, maxy
    poly_m = gpd.GeoSeries([poly], crs=4326).to_crs(3857).iloc[0]
    return poly_to_tiles(poly_m, tile_km, max_tiles)

@st.cache_resource(show_spinner=False)
def download_graph_bbox_cached(n, s, e, w, net_type):
    """Hàm tải có cache, gọi wrapper safe_graph_from_bbox"""
    return safe_graph_from_bbox(n, s, e, w, net_type)

def compose_graphs(graphs):
    """Gộp nhiều graph con thành một graph lớn"""
    valid_graphs = [g for g in graphs if g is not None and len(g) > 0]
    if not valid_graphs:
        return None
    
    # Compose trong NetworkX
    G_composed = nx.compose_all(valid_graphs)
    return G_composed

def compute_stats_for_graph(G):
    """Tính thống kê cơ bản"""
    # 1. Thống kê nodes/edges
    n = G.number_of_nodes()
    e = G.number_of_edges()
    
    # 2. Tính chiều dài
    # Lưu ý: G phải được project sang mét trước khi tính length
    s = safe_basic_stats(G)
    
    length_m = s.get("street_length_total", 0)
    length_km = float(length_m / 1000.0)
    
    return length_km, n, e

# ============================================================
# 4) LOGIC CHẠY CHÍNH
# ============================================================
go = st.button("🚀 Tải & Tính toán", type="primary")

if go:
    if st.session_state["busy"]:
        st.warning("Hệ thống đang bận. Vui lòng F5 nếu bị treo.")
        st.stop()
    st.session_state["busy"] = True

    try:
        # --------------------------------------------------------
        # XỬ LÝ INPUT
        # --------------------------------------------------------
        target_bboxes = [] # List các (n, s, e, w) cần tải

        if mode == "Địa danh (polygon)":
            if not place.strip():
                st.error("Vui lòng nhập tên địa danh.")
                st.session_state["busy"] = False
                st.stop()

            with st.spinner(f"Đang tìm kiếm '{place}'..."):
                try:
                    gdf_wgs, gdf_m, area_km2 = geocode_place_data(place)
                    st.success(f"Đã tìm thấy: Diện tích **{area_km2:,.1f} km²**")

                    # Quyết định: Tải 1 lần hay chia nhỏ?
                    if (not autosplit) or (area_km2 <= area_threshold_km2):
                        st.info("✅ Vùng nhỏ: Tải trực tiếp 1 lần.")
                        # Tải trực tiếp bằng place
                        with st.spinner("Đang tải dữ liệu mạng lưới..."):
                            # Dùng hàm của OSMnx (tự xử lý version bên trong thư viện)
                            if OX_MAJOR_VERSION >= 2:
                                G_raw = ox.graph.graph_from_place(place, network_type=network_type)
                            else:
                                G_raw = ox.graph_from_place(place, network_type=network_type)
                            
                            # Xử lý kết quả ngay tại đây (PROJECT GRAPH)
                            G_proj = safe_project_graph(G_raw)
                            km, nn, ne = compute_stats_for_graph(G_proj)
                            
                            st.metric("Tổng chiều dài đường", f"{km:,.2f} km")
                            st.write(f"Nodes: {nn} | Edges: {ne}")
                            
                            fig, ax = ox.plot.plot_graph(G_proj, show=False, close=True, node_size=0, edge_linewidth=0.5)
                            st.pyplot(fig)
                            st.session_state["busy"] = False
                            st.stop() # Kết thúc sớm cho trường hợp đơn giản
                    else:
                        st.warning(f"⚠️ Vùng lớn (> {area_threshold_km2} km²): Kích hoạt chia nhỏ (Tiling).")
                        with st.spinner("Đang chia lưới địa hình..."):
                            target_bboxes = poly_to_tiles(gdf_m.geometry.iloc[0], tile_km, max_tiles)
                
                except Exception as e:
                    # Bắt lỗi geocoding hoặc tải trực tiếp thất bại
                    st.error(f"Lỗi xử lý địa danh: {e}")
                    st.session_state["busy"] = False
                    st.stop()

        else: # Chế độ BBox
            st.write("Nhập toạ độ BBox (WGS84):")
            c1, c2, c3, c4 = st.columns(4)
            north = c1.number_input("North (Vĩ độ Bắc)", value=10.86, format="%.4f")
            south = c2.number_input("South (Vĩ độ Nam)", value=10.67, format="%.4f")
            east  = c3.number_input("East (Kinh độ Đông)", value=106.84, format="%.4f")
            west  = c4.number_input("West (Kinh độ Tây)", value=106.62, format="%.4f")

            if north <= south or east <= west:
                st.error("Toạ độ không hợp lệ (North > South, East > West).")
                st.session_state["busy"] = False
                st.stop()
            
            # Nếu autosplit bật, chia nhỏ bbox
            if autosplit:
                target_bboxes = bbox_to_tiles(north, south, east, west, tile_km, max_tiles)
            else:
                target_bboxes = [(north, south, east, west)]

        # --------------------------------------------------------
        # XỬ LÝ DOWNLOAD (TILING)
        # --------------------------------------------------------
        if not target_bboxes:
            st.error("Không tạo được ô lưới nào. Hãy kiểm tra lại toạ độ/địa danh.")
            st.session_state["busy"] = False
            st.stop()
        
        st.write(f"📋 **Kế hoạch:** Tải **{len(target_bboxes)}** ô lưới. Đang xử lý...")
        
        downloaded_graphs = []
        stats_rows = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Hàm worker cho thread pool
        def fetch_tile(idx, bbox_coords):
            n, s, e, w = bbox_coords
            try:
                # Gọi wrapper đã sửa lỗi
                G_sub = download_graph_bbox_cached(n, s, e, w, network_type)
                
                # Tính thống kê sơ bộ cho tile (cần project tạm để tính mét)
                if G_sub is not None and len(G_sub) > 0:
                    G_sub_proj = safe_project_graph(G_sub)
                    km_sub, nn_sub, ne_sub = compute_stats_for_graph(G_sub_proj)
                    return idx, G_sub, km_sub, nn_sub, ne_sub, (n, s, e, w), None
                else:
                    return idx, None, 0, 0, 0, (n, s, e, w), "Empty graph"
            except Exception as ex:
                return idx, None, 0, 0, 0, (n, s, e, w), str(ex)

        # Chạy tải dữ liệu
        results = []
        
        # 1. Chạy tuần tự (An toàn nhất để tránh rate limit)
        if concurrency == 1:
            for i, bbox in enumerate(target_bboxes):
                status_text.text(f"⏳ Đang tải ô {i+1}/{len(target_bboxes)}...")
                res = fetch_tile(i, bbox)
                results.append(res)
                progress_bar.progress((i + 1) / len(target_bboxes))
                time.sleep(delay_s) # Tôn trọng server
        
        # 2. Chạy song song (Nhanh nhưng dễ bị ban IP nếu quá nhanh)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(fetch_tile, i, bbox): i for i, bbox in enumerate(target_bboxes)}
                for i, fut in enumerate(as_completed(futures)):
                    results.append(fut.result())
                    progress_bar.progress((i + 1) / len(target_bboxes))
                    status_text.text(f"⏳ Đã xong {i + 1}/{len(target_bboxes)}...")

        # Xử lý kết quả
        for idx, G_sub, km, nn, ne, coords, err in results:
            if G_sub:
                downloaded_graphs.append(G_sub)
            
            stats_rows.append({
                "Tile ID": idx,
                "Length (km)": round(km, 3),
                "Nodes": nn,
                "Edges": ne,
                "Status": "OK" if not err else f"Error: {err}",
                "North": coords[0], "South": coords[1], "East": coords[2], "West": coords[3]
            })

        status_text.text("✅ Hoàn tất tải dữ liệu. Đang gộp đồ thị...")

        # --------------------------------------------------------
        # GỘP VÀ TÍNH TOÁN CUỐI CÙNG
        # --------------------------------------------------------
        if not downloaded_graphs:
            st.error("Không tải được dữ liệu đường nào (có thể khu vực này không có dữ liệu trên OSM).")
        else:
            G_final = compose_graphs(downloaded_graphs)
            
            if G_final is None or len(G_final) == 0:
                st.error("Đồ thị rỗng sau khi gộp.")
            else:
                # Project lần cuối để tính tổng chính xác
                with st.spinner("Đang xử lý hình học và tính toán tổng..."):
                    G_final_proj = safe_project_graph(G_final)
                    total_km, total_nodes, total_edges = compute_stats_for_graph(G_final_proj)

                # Hiển thị kết quả
                st.divider()
                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("🛣️ Tổng chiều dài", f"{total_km:,.2f} km")
                c_res2.metric("Nodes", f"{total_nodes:,}")
                c_res3.metric("Edges", f"{total_edges:,}")

                # Bảng chi tiết
                df_res = pd.DataFrame(stats_rows).sort_values("Tile ID")
                with st.expander("📄 Xem chi tiết từng ô lưới"):
                    st.dataframe(df_res, use_container_width=True)
                    csv = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Tải báo cáo CSV", csv, "road_network_stats.csv", "text/csv")

                # Vẽ bản đồ
                with st.spinner("Đang vẽ bản đồ (có thể lâu)..."):
                    fig, ax = ox.plot.plot_graph(
                        G_final_proj, 
                        show=False, 
                        close=True, 
                        node_size=0, 
                        edge_linewidth=0.5, 
                        edge_color="#333333",
                        bgcolor="white"
                    )
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi không mong muốn: {e}")
        st.exception(e)
    finally:
        st.session_state["busy"] = False
