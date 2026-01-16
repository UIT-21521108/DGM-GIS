
# app.py
# Streamlit app: Tính tổng chiều dài mạng lưới đường với OSMnx
# - Hỗ trợ: OSMnx v1.x và v2.x (tự động)
# - Chia lưới (tiling) cho vùng lớn, tôn trọng rate-limit
# - Project graph sang CRS phẳng để hết cảnh báo hình học

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
from shapely.geometry import box, Polygon

# Ẩn bớt warning lặt vặt (nếu muốn nhìn kỹ, hãy tắt dòng dưới)
warnings.filterwarnings("ignore")

# =========================
# OSMnx v1/v2 COMPAT SHIM
# =========================
# Alias theo namespace hiện có. Nếu thiếu (v1), rơi về API top-level cũ.
try:
    graph_from_place = ox.graph.graph_from_place
    graph_from_bbox  = ox.graph.graph_from_bbox
    basic_stats      = ox.stats.basic_stats
    plot_graph       = ox.plot.plot_graph
    geocode_to_gdf   = ox.geocoder.geocode_to_gdf
    project_graph_fn = ox.projection.project_graph
except AttributeError:
    # v1.x
    graph_from_place = getattr(ox, "graph_from_place", None) or getattr(ox.graph, "graph_from_place")
    graph_from_bbox  = getattr(ox, "graph_from_bbox", None)  or getattr(ox.graph, "graph_from_bbox")
    basic_stats      = getattr(ox, "basic_stats", None)
    plot_graph       = getattr(ox, "plot_graph", None)
    geocode_to_gdf   = getattr(ox, "geocode_to_gdf", None) or getattr(getattr(ox, "geocoder", object()), "geocode_to_gdf", None)
    project_graph_fn = getattr(ox, "project_graph", None) or getattr(getattr(ox, "projection", object()), "project_graph", None)

def safe_project_graph(G):
    return project_graph_fn(G) if project_graph_fn else G

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Độ dài mạng lưới đường (OSMnx)", page_icon="🗺️", layout="centered")

ox.settings.use_cache = True
ox.settings.log_console = False
try:
    ox.settings.overpass_rate_limit = True
except AttributeError:
    pass
ox.settings.timeout = 180

st.title("🗺️ Tính chiều dài đường (OSMnx Auto‑Compat)")
st.caption(f"OSMnx version: **{ox.__version__}**")

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
        index=2,  # mặc định "drive"
    )

with st.expander("⚙️ Tuỳ chọn nâng cao"):
    autosplit = st.checkbox("Tự chia nhỏ vùng lớn (Auto‑Tiling)", True)
    area_threshold_km2 = st.number_input("Ngưỡng kích hoạt chia nhỏ (km²)", 1.0, 10000.0, 100.0, 10.0)
    tile_km = st.slider("Kích thước ô lưới (km)", 1, 25, 5, 1)
    max_tiles = st.slider("Giới hạn số ô tối đa", 4, 400, 100, 4)
    delay_s = st.slider("Delay giữa các request (s)", 0.0, 5.0, 0.5, 0.1)
    concurrency = st.slider("Số luồng tải song song (thread)", 1, 5, 1, 1)
    show_tiles_outline = st.checkbox("Vẽ viền các ô lưới", True)

# =========================
# Hàm core
# =========================
@st.cache_data(show_spinner=False)
def geocode_place_data(place_name: str):
    gdf = geocode_to_gdf(place_name)  # 4326
    gdf_webm = gdf.to_crs(3857)       # mét
    area_km2 = float(gdf_webm.area.iloc[0] / 1e6)
    return gdf, gdf_webm, area_km2

def poly_to_tiles(poly_m: Polygon, tile_km: int, max_tiles: int):
    # Chia theo lưới ô vuông đơn giản
    minx, miny, maxx, maxy = poly_m.bounds
    step = tile_km * 1000

    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)

    bboxes = []
    for x in xs:
        for y in ys:
            cell = box(x, y, x + step, y + step)
            if not poly_m.intersects(cell):
                continue
            inter = poly_m.intersection(cell)
            if inter.is_empty:
                continue
            inter_wgs = gpd.GeoSeries([inter], crs=3857).to_crs(4326).iloc[0]
            lon_min, lat_min, lon_max, lat_max = inter_wgs.bounds
            bboxes.append((lat_max, lat_min, lon_max, lon_min))  # N, S, E, W
            if len(bboxes) >= max_tiles:
                return bboxes
    return bboxes

def bbox_to_tiles(n: float, s: float, e: float, w: float, tile_km: int, max_tiles: int):
    poly = box(w, s, e, n)
    poly_m = gpd.GeoSeries([poly], crs=4326).to_crs(3857).iloc[0]
    return poly_to_tiles(poly_m, tile_km, max_tiles)

@st.cache_resource(show_spinner=False)
def download_graph_bbox_cached(n: float, s: float, e: float, w: float, net_type: str):
    # v2: graph.graph_from_bbox(n, s, e, w, ...)
    # v1: graph_from_bbox(n, s, e, w, ...)
    return graph_from_bbox(n, s, e, w, network_type=net_type)

def compose_graphs(graphs):
    graphs = [g for g in graphs if g is not None and len(g) > 0]
    if not graphs:
        return None
    return nx.compose_all(graphs)

def compute_stats_for_graph(G):
    # G nên là graph đã project sang CRS phẳng
    s = basic_stats(G)  # an toàn cho v1/v2 (không truyền clean_int_tol)
    length_m = s.get("street_length_total", 0.0)
    return float(length_m / 1000.0), G.number_of_nodes(), G.number_of_edges()

# =========================
# Chạy
# =========================
go = st.button("🚀 Tải & Tính toán", type="primary")

if go:
    if st.session_state["busy"]:
        st.warning("Hệ thống đang bận. Vui lòng đợi tác vụ trước hoàn tất.")
        st.stop()
    st.session_state["busy"] = True

    try:
        target_bboxes = []

        # ---------- MODE: PLACE ----------
        if mode == "Địa danh (polygon)":
            if not place.strip():
                st.error("Vui lòng nhập tên địa danh.")
                st.session_state["busy"] = False
                st.stop()

            with st.spinner(f"Geocoding '{place}'..."):
                gdf_wgs, gdf_m, area_km2 = geocode_place_data(place)
            st.success(f"Diện tích ước lượng: **{area_km2:,.1f} km²**")

            if (not autosplit) or (area_km2 <= area_threshold_km2):
                st.info("✅ Vùng nhỏ → tải trực tiếp 1 lần.")
                with st.spinner("Đang tải dữ liệu mạng lưới..."):
                    G_raw = graph_from_place(place, network_type=network_type)
                    G_proj = safe_project_graph(G_raw)  # project sang CRS phẳng
                    km, nn, ne = compute_stats_for_graph(G_proj)

                st.metric("Tổng chiều dài đường", f"{km:,.2f} km")
                st.write(f"Nodes: {nn:,} | Edges: {ne:,}")

                fig, ax = plot_graph(G_proj, show=False, close=True, node_size=0, edge_linewidth=0.5)
                st.pyplot(fig)

                st.session_state["busy"] = False
                st.stop()

            else:
                st.warning(f"⚠️ Vùng lớn (> {area_threshold_km2} km²) → kích hoạt chia lưới.")
                with st.spinner("Đang tạo lưới ô..."):
                    target_bboxes = poly_to_tiles(gdf_m.geometry.iloc[0], tile_km, max_tiles)

        # ---------- MODE: BBOX ----------
        else:
            st.write("Nhập toạ độ BBox (WGS84):")
            c1, c2, c3, c4 = st.columns(4)
            north = c1.number_input("North (lat)", value=10.86, format="%.4f")
            south = c2.number_input("South (lat)", value=10.67, format="%.4f")
            east  = c3.number_input("East (lon)",  value=106.84, format="%.4f")
            west  = c4.number_input("West (lon)",  value=106.62, format="%.4f")

            if north <= south or east <= west:
                st.error("Toạ độ không hợp lệ (North > South, East > West).")
                st.session_state["busy"] = False
                st.stop()

            target_bboxes = bbox_to_tiles(north, south, east, west, tile_km, max_tiles) if autosplit else [(north, south, east, west)]

        # ---------- TẢI TỪNG TILE ----------
        if not target_bboxes:
            st.error("Không tạo được ô lưới nào. Kiểm tra địa danh/toạ độ.")
            st.session_state["busy"] = False
            st.stop()

        st.write(f"📋 Kế hoạch: tải **{len(target_bboxes)}** ô lưới ...")

        downloaded_graphs = []
        rows = []
        progress = st.progress(0)
        status = st.empty()

        def fetch_tile(idx_bbox):
            idx, (n, s, e, w) = idx_bbox
            try:
                Gi = download_graph_bbox_cached(n, s, e, w, network_type)
                if Gi and len(Gi) > 0:
                    Gip = safe_project_graph(Gi)
                    km, nn, ne = compute_stats_for_graph(Gip)
                    return idx, Gi, km, nn, ne, None
                return idx, None, 0.0, 0, 0, "Empty graph"
            except Exception as ex:
                return idx, None, 0.0, 0, 0, str(ex)

        results = []
        if concurrency == 1:
            for i, bb in enumerate(target_bboxes, 1):
                status.text(f"⏳ Đang tải ô {i}/{len(target_bboxes)} ...")
                results.append(fetch_tile((i, bb)))
                progress.progress(i / len(target_bboxes))
                time.sleep(delay_s)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = {pool.submit(fetch_tile, (i, bb)): i for i, bb in enumerate(target_bboxes, 1)}
                done = 0
                for fut in as_completed(futs):
                    results.append(fut.result())
                    done += 1
                    progress.progress(done / len(target_bboxes))
                    status.text(f"⏳ Đã xong {done}/{len(target_bboxes)} ...")
                    time.sleep(delay_s)

        for i, Gi, km, nn, ne, err in sorted(results, key=lambda x: x[0]):
            if Gi:
                downloaded_graphs.append(Gi)
            rows.append({
                "Tile ID": i,
                "Length (km)": round(km, 3),
                "Nodes": nn,
                "Edges": ne,
                "Status": "OK" if not err else f"Error: {err}"
            })

        status.text("✅ Hoàn tất tải. Đang gộp đồ thị ...")

        # ---------- GHÉP & TÍNH TỔNG ----------
        if not downloaded_graphs:
            st.error("Không tải được dữ liệu đường nào.")
        else:
            G = compose_graphs(downloaded_graphs)
            if not G or len(G) == 0:
                st.error("Đồ thị rỗng sau khi gộp.")
            else:
                with st.spinner("Đang project & tính toán tổng ..."):
                    Gp = safe_project_graph(G)
                    total_km, total_nodes, total_edges = compute_stats_for_graph(Gp)

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("🛣️ Tổng chiều dài", f"{total_km:,.2f} km")
                c2.metric("Nodes", f"{total_nodes:,}")
                c3.metric("Edges", f"{total_edges:,}")

                df = pd.DataFrame(rows)
                with st.expander("📄 Chi tiết từng tile"):
                    st.dataframe(df, use_container_width=True)
                    st.download_button("⬇️ Tải CSV", df.to_csv(index=False).encode("utf-8"),
                                       "road_network_stats.csv", "text/csv")

                fig, ax = plot_graph(Gp, show=False, close=True, node_size=0, edge_linewidth=0.5, edge_color="#333", bgcolor="white")
                if show_tiles_outline:
                    # Vẽ khung tile theo WGS bbox (chỉ minh hoạ, không chính xác CRS của ax)
                    for (n, s, e, w) in target_bboxes:
                        xs, ys = [w, e, e, w, w], [s, s, n, n, s]
                        try:
                            ax.plot(xs, ys, "r-", linewidth=0.8, alpha=0.6)
                        except Exception:
                            pass
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi không mong muốn: {e}")
        st.exception(e)
    finally:
        st.session_state["busy"] = False
