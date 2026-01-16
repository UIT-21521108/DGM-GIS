# app.py
# Streamlit app: Tính tổng chiều dài mạng lưới đường (KMU) từ OpenStreetMap bằng OSMnx
# - Hỗ trợ Địa danh (polygon) & BBox (tọa độ)
# - Tự chia nhỏ vùng lớn thành lưới (tiles) để tải tuần tự / có kiểm soát
# - ĐÃ THÊM "OSMnx v1/v2 compatibility shim" để chạy được trên cả 1.x và 2.x

from __future__ import annotations

import time
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Polygon

# =========================
# OSMnx v1/v2 COMPATIBILITY SHIM
# =========================
# OSMnx v2.0 chuyển API sang namespace mới:
#   - graph_from_place  -> ox.graph.graph_from_place
#   - graph_from_bbox   -> ox.graph.graph_from_bbox
#   - basic_stats       -> ox.stats.basic_stats
#   - plot_graph        -> ox.plot.plot_graph
#   - geocode_to_gdf    -> ox.geocoder.geocode_to_gdf
# Tài liệu v2: https://osmnx.readthedocs.io/en/stable/osmnx.html
# Changelog: OSMnx v2.0 thêm hỗ trợ Python 3.13, đổi namespace, v.v.
#            https://github.com/gboeing/osmnx/blob/main/CHANGELOG.md
try:
    graph_from_place = ox.graph.graph_from_place
    graph_from_bbox  = ox.graph.graph_from_bbox
    basic_stats      = ox.stats.basic_stats
    plot_graph       = ox.plot.plot_graph
    geocode_to_gdf   = ox.geocoder.geocode_to_gdf
except AttributeError:
    # Fallback cho OSMnx v1.x (API cũ)
    graph_from_place = ox.graph_from_place
    graph_from_bbox  = ox.graph_from_bbox
    basic_stats      = ox.basic_stats
    plot_graph       = ox.plot_graph
    geocode_to_gdf   = ox.geocode_to_gdf
# =========================

# =========================
# CẤU HÌNH CHUNG
# =========================
st.set_page_config(
    page_title="Độ dài mạng lưới đường (OSMnx) — Chia lưới vùng lớn",
    page_icon="🗺️",
    layout="centered",
)

# OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.overpass_rate_limit = True  # tự chờ nếu bị rate-limit
ox.settings.timeout = 180               # tăng timeout cho truy vấn lớn

st.title("🗺️ Tính tổng chiều dài mạng lưới đường (KMU) — Tối ưu vùng lớn")
st.markdown(
    "Nhập **địa danh (polygon)** hoặc **BBox (tọa độ)**. "
    "Nếu khu vực quá lớn, app sẽ **tự chia lưới** và tải lần lượt để tránh timeout."
)

if "busy" not in st.session_state:
    st.session_state["busy"] = False
if "place_text" not in st.session_state:
    st.session_state["place_text"] = "Ho Chi Minh City, Vietnam"

# =========================
# INPUTS
# =========================
mode = st.radio("Chế độ nhập khu vực", ["Địa danh (polygon)", "BBox (tọa độ)"], horizontal=True)

col_top = st.columns([2, 1])
with col_top[0]:
    if mode == "Địa danh (polygon)":
        preset = st.selectbox(
            "Preset địa danh",
            options=[
                "— (Không dùng preset) —",
                "District 1, Ho Chi Minh City, Vietnam",
                "Thu Duc City, Ho Chi Minh City, Vietnam",
                "Hue, Vietnam",
                "Hai Chau District, Danang, Vietnam",
                "Son Tra District, Danang, Vietnam",
                "Singapore",
            ],
            index=0,
        )
        if preset != "— (Không dùng preset) —":
            st.session_state["place_text"] = preset
        place = st.text_input("Địa danh (place):", key="place_text")
    else:
        place = ""

with col_top[1]:
    network_type = st.selectbox(
        "Loại đường",
        options=["all", "all_public", "drive", "drive_service", "walk", "bike"],
        index=0,
        help="Chọn loại mạng lưới muốn tải."
    )

with st.expander("⚙️ Tuỳ chọn nâng cao cho vùng lớn"):
    autosplit = st.checkbox("Tự động chia nhỏ nếu vùng lớn", value=True)
    area_threshold_km2 = st.number_input(
        "Ngưỡng diện tích coi là 'vùng lớn' (km²)", min_value=1.0, max_value=5000.0,
        value=120.0, step=10.0,
        help="Nếu diện tích lớn hơn ngưỡng này, app sẽ bật chia lưới."
    )
    tile_km = st.slider(
        "Kích thước mỗi ô lưới (km)", min_value=2, max_value=25, value=8, step=1,
        help="Ô nhỏ: nhiều request hơn; ô lớn quá: dễ timeout."
    )
    max_tiles = st.slider(
        "Số ô tối đa", min_value=4, max_value=400, value=120, step=4,
        help="Giới hạn để tránh gửi quá nhiều request."
    )
    delay_s = st.slider(
        "Thời gian nghỉ giữa các ô (giây)", min_value=0.0, max_value=2.0, value=0.5, step=0.1,
        help="Giúp tôn trọng rate‑limit Overpass."
    )
    concurrency = st.slider(
        "Mức song song khi tải tiles (1 = tuần tự an toàn)", min_value=1, max_value=3, value=1, step=1,
        help="Tăng lên 2–3 có thể nhanh hơn nhưng dễ chạm rate‑limit. Khuyến nghị: 1."
    )
    show_tiles_outline = st.checkbox("Vẽ lưới tiles chồng lên đồ thị", value=False)

# =========================
# HÀM CORE
# =========================
@st.cache_data(show_spinner=False)
def geocode_place(place_name: str):
    gdf = geocode_to_gdf(place_name)   # 4326
    gdf_webm = gdf.to_crs(3857)        # mét
    area_km2 = float(gdf_webm.area.iloc[0] / 1e6)
    return gdf, gdf_webm, area_km2

def poly_to_tiles(polygon_m: Polygon, tile_km: int, max_tiles: int) -> List[Tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = polygon_m.bounds
    step = tile_km * 1000
    xs = np.arange(minx, maxx + step, step)
    ys = np.arange(miny, maxy + step, step)
    bboxes = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cell = box(xs[i], ys[j], xs[i + 1], ys[j + 1])
            inter = polygon_m.intersection(cell)
            if inter.is_empty:
                continue
            inter_wgs = gpd.GeoSeries([inter], crs=3857).to_crs(4326).iloc[0]
            lon_min, lat_min, lon_max, lat_max = inter_wgs.bounds
            bboxes.append((lat_max, lat_min, lon_max, lon_min))  # (N, S, E, W)
            if len(bboxes) >= max_tiles:
                return bboxes
    return bboxes

def bbox_to_tiles(north: float, south: float, east: float, west: float, tile_km: int, max_tiles: int):
    poly_wgs = box(west, south, east, north)
    poly_m = gpd.GeoSeries([poly_wgs], crs=4326).to_crs(3857).iloc[0]
    return poly_to_tiles(poly_m, tile_km, max_tiles)

@st.cache_resource(show_spinner=False)
def download_graph_bbox(north: float, south: float, east: float, west: float, net_type: str):
    return graph_from_bbox(north, south, east, west, network_type=net_type)

def compose_graphs(graphs: List[nx.MultiDiGraph]) -> Optional[nx.MultiDiGraph]:
    graphs = [g for g in graphs if g is not None]
    if not graphs:
        return None
    G = graphs[0]
    for Gi in graphs[1:]:
        G = nx.compose(G, Gi)
    return G

def compute_stats(G: nx.MultiDiGraph) -> dict:
    s = basic_stats(G, clean_int_tol=15)
    s["street_length_total_km"] = float(s.get("street_length_total", 0.0) / 1000.0)
    return s

def compute_tile_stats(Gi: nx.MultiDiGraph) -> tuple[float, int, int]:
    s = basic_stats(Gi, clean_int_tol=15)
    km = float(s.get("street_length_total", 0.0) / 1000.0)
    return km, Gi.number_of_nodes(), Gi.number_of_edges()

# =========================
# NÚT CHẠY
# =========================
go = st.button("Tải & Tính toán", type="primary")

if go:
    if st.session_state["busy"]:
        st.warning("Hệ thống đang xử lý yêu cầu trước. Vui lòng đợi xong rồi chạy tiếp.")
        st.stop()
    st.session_state["busy"] = True

    try:
        # ---- 1) PLACE MODE ----
        if mode == "Địa danh (polygon)":
            if not place.strip():
                st.error("Vui lòng nhập địa danh hợp lệ.")
                st.stop()

            with st.spinner("Geocoding & ước lượng diện tích…"):
                gdf_wgs, gdf_m, area_km2 = geocode_place(place)
            st.caption(f"Diện tích ước lượng: **{area_km2:,.1f} km²**")

            if (not autosplit) or (area_km2 <= area_threshold_km2):
                st.info("Vùng nhỏ hoặc không bật chia lưới → tải trực tiếp theo polygon.")
                with st.spinner("Đang tải từ Overpass…"):
                    G = graph_from_place(place, network_type=network_type)
                stats = compute_stats(G)
                st.success(f"✅ Tổng chiều dài (KMU): **{stats['street_length_total_km']:,.3f} km**")
                fig, ax = plot_graph(G, show=False, close=False, node_size=0, edge_linewidth=0.8, bgcolor="white")
                st.pyplot(fig, clear_figure=True)

            else:
                st.warning("Khu vực lớn → **bật chia lưới** để tải tuần tự/có kiểm soát.")
                with st.spinner("Đang tạo lưới tiles theo polygon…"):
                    bboxes = poly_to_tiles(gdf_m.geometry.iloc[0], tile_km=tile_km, max_tiles=max_tiles)

                if not bboxes:
                    st.error("Không tạo được ô nào giao với polygon. Hãy tăng kích thước ô hoặc kiểm tra place.")
                    st.stop()

                graphs, rows = [], []
                progress = st.progress(0, text="Bắt đầu tải từng ô…")
                status = st.empty()

                def fetch_one(idx_bbox):
                    idx, (n, s, e, w) = idx_bbox
                    Gi = download_graph_bbox(n, s, e, w, network_type)
                    km, nn, ne = compute_tile_stats(Gi)
                    return idx, Gi, km, nn, ne, (n, s, e, w)

                if concurrency == 1:
                    for idx, bb in enumerate(bboxes, start=1):
                        status.text(f"Đang tải ô {idx}/{len(bboxes)}")
                        try:
                            _, Gi, km, nn, ne, (n, s, e, w) = fetch_one((idx, bb))
                            graphs.append(Gi)
                            rows.append({"tile_id": idx, "north": n, "south": s, "east": e, "west": w,
                                         "street_km": km, "nodes": nn, "edges": ne})
                        except Exception as ex:
                            st.warning(f"Ô {idx} lỗi: {ex}")
                        time.sleep(delay_s)
                        progress.progress(idx / len(bboxes))
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futures = {pool.submit(fetch_one, (i, bb)): i for i, bb in enumerate(bboxes, 1)}
                        done = 0
                        for fut in as_completed(futures):
                            i = futures[fut]
                            try:
                                _, Gi, km, nn, ne, (n, s, e, w) = fut.result()
                                graphs.append(Gi)
                                rows.append({"tile_id": i, "north": n, "south": s, "east": e, "west": w,
                                             "street_km": km, "nodes": nn, "edges": ne})
                            except Exception as ex:
                                st.warning(f"Ô {i} lỗi: {ex}")
                            done += 1
                            progress.progress(done / len(bboxes))
                            status.text(f"Đã xong {done}/{len(bboxes)} ô…")
                            time.sleep(delay_s)

                status.text("Đang gộp các ô…")
                G = compose_graphs(graphs)
                if G is None:
                    st.error("Không tải được bất kỳ ô nào.")
                    st.stop()

                stats = compute_stats(G)
                st.success(f"✅ Tổng chiều dài (KMU): **{stats['street_length_total_km']:,.3f} km**")

                tiles_df = pd.DataFrame(rows).sort_values("tile_id")
                st.dataframe(tiles_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Tải CSV theo tile", tiles_df.to_csv(index=False).encode("utf-8"),
                                   "tile_stats.csv", "text/csv")

                fig, ax = plot_graph(G, show=False, close=False, node_size=0, edge_linewidth=0.8, bgcolor="white")
                if show_tiles_outline:
                    for (n, s, e, w) in bboxes:
                        xs, ys = [w, e, e, w, w], [s, s, n, n, s]
                        ax.plot(xs, ys, color="red", linewidth=0.8, alpha=0.5)
                st.pyplot(fig, clear_figure=True)

        # ---- 2) BBOX TOẠ ĐỘ ----
        else:
            st.markdown("Nhập **tọa độ WGS84** (độ):")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                north = st.number_input("North (lat)", value=10.86, format="%.6f")
            with c2:
                south = st.number_input("South (lat)", value=10.67, format="%.6f")
            with c3:
                east = st.number_input("East (lon)", value=106.84, format="%.6f")
            with c4:
                west = st.number_input("West (lon)", value=106.62, format="%.6f")

            if north <= south or east <= west:
                st.error("BBox không hợp lệ: cần north>south và east>west.")
                st.stop()

            if autosplit:
                bboxes = bbox_to_tiles(north, south, east, west, tile_km=tile_km, max_tiles=max_tiles)
                st.write(f"Số ô sẽ tải: **{len(bboxes)}** (mỗi ô nghỉ {delay_s}s; song song: {concurrency})")
            else:
                bboxes = [(north, south, east, west)]

            graphs, rows = [], []
            progress = st.progress(0, text="Bắt đầu tải từng ô…")
            status = st.empty()

            def fetch_one(idx_bbox):
                idx, (n, s, e, w) = idx_bbox
                Gi = download_graph_bbox(n, s, e, w, network_type)
                km, nn, ne = compute_tile_stats(Gi)
                return idx, Gi, km, nn, ne, (n, s, e, w)

            if concurrency == 1:
                for idx, bb in enumerate(bboxes, start=1):
                    status.text(f"Đang tải ô {idx}/{len(bboxes)}")
                    try:
                        _, Gi, km, nn, ne, (n, s, e, w) = fetch_one((idx, bb))
                        graphs.append(Gi)
                        rows.append({"tile_id": idx, "north": n, "south": s, "east": e, "west": w,
                                     "street_km": km, "nodes": nn, "edges": ne})
                    except Exception as ex:
                        st.warning(f"Ô {idx} lỗi: {ex}")
                    time.sleep(delay_s)
                    progress.progress(idx / len(bboxes))
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = {pool.submit(fetch_one, (i, bb)): i for i, bb in enumerate(bboxes, 1)}
                    done = 0
                    for fut in as_completed(futures):
                        i = futures[fut]
                        try:
                            _, Gi, km, nn, ne, (n, s, e, w) = fut.result()
                            graphs.append(Gi)
                            rows.append({"tile_id": i, "north": n, "south": s, "east": e, "west": w,
                                         "street_km": km, "nodes": nn, "edges": ne})
                        except Exception as ex:
                            st.warning(f"Ô {i} lỗi: {ex}")
                        done += 1
                        progress.progress(done / len(bboxes))
                        status.text(f"Đã xong {done}/{len(bboxes)} ô…")
                        time.sleep(delay_s)

            status.text("Đang gộp các ô…")
            G = compose_graphs(graphs)
            if G is None:
                st.error("Không tải được bất kỳ ô nào.")
                st.stop()

            stats = compute_stats(G)
            st.success(f"✅ Tổng chiều dài (KMU): **{stats['street_length_total_km']:,.3f} km**")

            tiles_df = pd.DataFrame(rows).sort_values("tile_id")
            st.dataframe(tiles_df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Tải CSV theo tile", tiles_df.to_csv(index=False).encode("utf-8"),
                               "tile_stats.csv", "text/csv")

            fig, ax = plot_graph(G, show=False, close=False, node_size=0, edge_linewidth=0.8, bgcolor="white")
            if show_tiles_outline:
                for (n, s, e, w) in bboxes:
                    xs, ys = [w, e, e, w, w], [s, s, n, n, s]
                    ax.plot(xs, ys, color="red", linewidth=0.8, alpha=0.5)
            st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.error("Có lỗi xảy ra trong quá trình tải/ghép/hiển thị dữ liệu.")
        st.exception(e)
    finally:
        st.session_state["busy"] = False
else:
    st.info("Chọn chế độ, nhập khu vực, điều chỉnh tham số nếu cần → bấm **Tải & Tính toán**.")
