
# app.py
# Streamlit app: Tính tổng chiều dài mạng lưới đường với OSMnx
# - Hỗ trợ Địa danh & BBox
# - Tối ưu vùng lớn bằng chia lưới
# - ĐÃ THÊM compat-shim cho OSMnx 1.x/2.x
# - ĐÃ THÊM project_graph để tránh warning CRS (buffer/centroid)

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

# ============================================================
# 1) OSMnx v1/v2 COMPAT SHIM  – tránh lỗi API khi dùng OSMnx 2
# ============================================================
try:
    graph_from_place = ox.graph.graph_from_place
    graph_from_bbox  = ox.graph.graph_from_bbox
    basic_stats      = ox.stats.basic_stats
    plot_graph       = ox.plot.plot_graph
    geocode_to_gdf   = ox.geocoder.geocode_to_gdf
except AttributeError:
    graph_from_place = ox.graph_from_place
    graph_from_bbox  = ox.graph_from_bbox
    basic_stats      = ox.basic_stats
    plot_graph       = ox.plot_graph
    geocode_to_gdf   = ox.geocode_to_gdf

# ============================================================
# 2) Streamlit UI
# ============================================================
st.set_page_config(
    page_title="Độ dài mạng lưới đường (OSMnx)",
    page_icon="🗺️",
    layout="centered",
)

ox.settings.use_cache = True
ox.settings.log_console = False
ox.settings.overpass_rate_limit = True
ox.settings.timeout = 180

st.title("🗺️ Tính tổng chiều dài mạng lưới đường (KMU) — Tối ưu vùng lớn")

if "busy" not in st.session_state:
    st.session_state["busy"] = False
if "place_text" not in st.session_state:
    st.session_state["place_text"] = "Ho Chi Minh City, Vietnam"

mode = st.radio("Chọn chế độ nhập:", ["Địa danh (polygon)", "BBox"], horizontal=True)

colA, colB = st.columns([2, 1])
with colA:
    if mode == "Địa danh (polygon)":
        preset = st.selectbox(
            "Preset",
            [
                "—",
                "District 1, Ho Chi Minh City, Vietnam",
                "Thu Duc City, Ho Chi Minh City, Vietnam",
                "Hue, Vietnam",
                "Hai Chau District, Danang, Vietnam",
                "Son Tra District, Danang, Vietnam",
                "Singapore",
            ],
            index=0,
        )
        if preset != "—":
            st.session_state["place_text"] = preset
        place = st.text_input("Địa danh:", key="place_text")
    else:
        place = ""

with colB:
    network_type = st.selectbox(
        "Loại đường",
        ["all", "all_public", "drive", "drive_service", "walk", "bike"],
        index=0,
    )

with st.expander("⚙️ Tuỳ chọn nâng cao"):
    autosplit = st.checkbox("Tự chia nhỏ vùng lớn", True)
    area_threshold_km2 = st.number_input("Ngưỡng vùng lớn (km²)", 1.0, 5000.0, 120.0, 10.0)
    tile_km = st.slider("Kích thước ô lưới (km)", 2, 25, 8, 1)
    max_tiles = st.slider("Số ô tối đa", 4, 400, 120, 4)
    delay_s = st.slider("Delay mỗi ô (s)", 0.0, 2.0, 0.5, 0.1)
    concurrency = st.slider("Mức song song (1 an toàn)", 1, 3, 1, 1)
    show_tiles_outline = st.checkbox("Vẽ viền tile", False)

# ============================================================
# 3) HÀM CORE – đã tối ưu
# ============================================================
@st.cache_data(show_spinner=False)
def geocode_place(place_name: str):
    gdf = geocode_to_gdf(place_name)
    gdf_webm = gdf.to_crs(3857)
    area_km2 = float(gdf_webm.area.iloc[0] / 1e6)
    return gdf, gdf_webm, area_km2

def poly_to_tiles(poly_m: Polygon, tile_km, max_tiles):
    minx, miny, maxx, maxy = poly_m.bounds
    step = tile_km * 1000

    xs = np.arange(minx, maxx + step, step)
    ys = np.arange(miny, maxy + step, step)

    bboxes = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cell = box(xs[i], ys[j], xs[i+1], ys[j+1])
            inter = poly_m.intersection(cell)
            if inter.is_empty:
                continue
            inter_wgs = gpd.GeoSeries([inter], crs=3857).to_crs(4326).iloc[0]
            lon_min, lat_min, lon_max, lat_max = inter_wgs.bounds
            bboxes.append((lat_max, lat_min, lon_max, lon_min))
            if len(bboxes) >= max_tiles:
                break
    return bboxes

def bbox_to_tiles(n, s, e, w, tile_km, max_tiles):
    poly = box(w, s, e, n)
    poly_m = gpd.GeoSeries([poly], crs=4326).to_crs(3857).iloc[0]
    return poly_to_tiles(poly_m, tile_km, max_tiles)

@st.cache_resource(show_spinner=False)
def download_graph_bbox(n, s, e, w, net_type):
    return graph_from_bbox(n, s, e, w, network_type=net_type)

def compose_graphs(graphs):
    graphs = [g for g in graphs if g is not None]
    if not graphs:
        return None
    G = graphs[0]
    for Gi in graphs[1:]:
        G = nx.compose(G, Gi)
    return G

def compute_stats(G):
    s = basic_stats(G, clean_int_tol=15)
    s["street_length_total_km"] = float(s["street_length_total"] / 1000.0)
    return s

def compute_tile_stats(Gi):
    s = basic_stats(Gi, clean_int_tol=15)
    return float(s["street_length_total"] / 1000.0), Gi.number_of_nodes(), Gi.number_of_edges()

# ============================================================
# 4) NÚT CHẠY
# ============================================================
go = st.button("Tải & Tính toán", type="primary")

if go:
    if st.session_state["busy"]:
        st.warning("Đang chạy tác vụ khác, vui lòng đợi.")
        st.stop()
    st.session_state["busy"] = True

    try:
        # ====================================================
        # A) ĐỊA DANH
        # ====================================================
        if mode == "Địa danh (polygon)":

            if not place.strip():
                st.error("Bạn chưa nhập địa danh.")
                st.stop()

            with st.spinner("Geocoding..."):
                gdf_wgs, gdf_m, area_km2 = geocode_place(place)
            st.caption(f"Diện tích: **{area_km2:.1f} km²**")

            # Vùng nhỏ → tải trực tiếp
            if (not autosplit) or (area_km2 <= area_threshold_km2):
                st.info("Vùng nhỏ → tải trực tiếp")

                G = graph_from_place(place, network_type=network_type)

                # NEW: PROJECT GRAPH để hết WARNING
                Gp = ox.projection.project_graph(G)

                stats = compute_stats(Gp)
                st.success(f"Tổng chiều dài: **{stats['street_length_total_km']:.3f} km**")

                fig, ax = plot_graph(Gp, show=False, close=False, node_size=0, edge_linewidth=0.8)
                st.pyplot(fig, clear_figure=True)

            # Vùng lớn → chia lưới
            else:
                st.warning("Vùng lớn → chia lưới")

                with st.spinner("Đang chia lưới..."):
                    bboxes = poly_to_tiles(gdf_m.geometry.iloc[0], tile_km, max_tiles)
                if not bboxes:
                    st.error("Không tạo được tile.")
                    st.stop()

                graphs, rows = [], []
                progress = st.progress(0)
                status = st.empty()

                def fetch(idx_bbox):
                    idx, (n, s, e, w) = idx_bbox
                    Gi = download_graph_bbox(n, s, e, w, network_type)
                    km, nn, ne = compute_tile_stats(Gi)
                    return idx, Gi, km, nn, ne, (n, s, e, w)

                # Tuần tự
                if concurrency == 1:
                    for idx, bb in enumerate(bboxes, 1):
                        status.text(f"Tải ô {idx}/{len(bboxes)}...")
                        try:
                            _, Gi, km, nn, ne, coords = fetch((idx, bb))
                            graphs.append(Gi)
                            rows.append({
                                "tile_id": idx,
                                "north": coords[0], "south": coords[1],
                                "east": coords[2], "west": coords[3],
                                "street_km": km, "nodes": nn, "edges": ne,
                            })
                        except Exception as ex:
                            st.warning(f"Lỗi tile {idx}: {ex}")
                        time.sleep(delay_s)
                        progress.progress(idx / len(bboxes))

                # Song song
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futs = {pool.submit(fetch, (i, bb)): i for i, bb in enumerate(bboxes, 1)}
                        done = 0
                        for fut in as_completed(futs):
                            i = futs[fut]
                            try:
                                _, Gi, km, nn, ne, coords = fut.result()
                                graphs.append(Gi)
                                rows.append({
                                    "tile_id": i,
                                    "north": coords[0], "south": coords[1],
                                    "east": coords[2], "west": coords[3],
                                    "street_km": km, "nodes": nn, "edges": ne,
                                })
                            except Exception as ex:
                                st.warning(f"Lỗi tile {i}: {ex}")
                            done += 1
                            progress.progress(done / len(bboxes))
                            status.text(f"Đã xong {done}/{len(bboxes)}")

                # GHÉP TILE
                G = compose_graphs(graphs)
                if G is None:
                    st.error("Không tải được bất kỳ tile nào.")
                    st.stop()

                # NEW: PROJECT GRAPH
                Gp = ox.projection.project_graph(G)

                stats = compute_stats(Gp)
                st.success(f"KMU: **{stats['street_length_total_km']:.3f} km**")

                df = pd.DataFrame(rows).sort_values("tile_id")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode()
                st.download_button("⬇️ Tải CSV", csv, "tile_stats.csv", "text/csv")

                fig, ax = plot_graph(Gp, show=False, close=False, node_size=0, edge_linewidth=0.8)
                if show_tiles_outline:
                    for row in rows:
                        n, s, e, w = row["north"], row["south"], row["east"], row["west"]
                        xs = [w, e, e, w, w]
                        ys = [s, s, n, n, s]
                        ax.plot(xs, ys, "r-", linewidth=0.8)
                st.pyplot(fig, clear_figure=True)

        # ====================================================
        # B) BBOX
        # ====================================================
        else:
            st.write("Nhập BBox (WGS84)")

            c1, c2, c3, c4 = st.columns(4)
            with c1: north = st.number_input("North", value=10.86)
            with c2: south = st.number_input("South", value=10.67)
            with c3: east  = st.number_input("East", value=106.84)
            with c4: west  = st.number_input("West", value=106.62)

            if north <= south or east <= west:
                st.error("BBox không hợp lệ.")
                st.stop()

            bboxes = bbox_to_tiles(north, south, east, west, tile_km, max_tiles) if autosplit else [(north, south, east, west)]

            graphs, rows = [], []
            progress = st.progress(0)
            status = st.empty()

            def fetch(idx_bbox):
                idx, (n, s, e, w) = idx_bbox
                Gi = download_graph_bbox(n, s, e, w, network_type)
                km, nn, ne = compute_tile_stats(Gi)
                return idx, Gi, km, nn, ne, (n, s, e, w)

            if concurrency == 1:
                for idx, bb in enumerate(bboxes, 1):
                    status.text(f"Tải ô {idx}/{len(bboxes)}...")
                    try:
                        _, Gi, km, nn, ne, coords = fetch((idx, bb))
                        graphs.append(Gi)
                        rows.append({
                            "tile_id": idx,
                            "north": coords[0], "south": coords[1],
                            "east": coords[2], "west": coords[3],
                            "street_km": km, "nodes": nn, "edges": ne,
                        })
                    except Exception as ex:
                        st.warning(f"Lỗi tile {idx}: {ex}")
                    time.sleep(delay_s)
                    progress.progress(idx / len(bboxes))

            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futs = {pool.submit(fetch, (i, bb)): i for i, bb in enumerate(bboxes, 1)}
                    done = 0
                    for fut in as_completed(futs):
                        i = futs[fut]
                        try:
                            _, Gi, km, nn, ne, coords = fut.result()
                            graphs.append(Gi)
                            rows.append({
                                "tile_id": i,
                                "north": coords[0], "south": coords[1],
                                "east": coords[2], "west": coords[3],
                                "street_km": km, "nodes": nn, "edges": ne,
                            })
                        except Exception as ex:
                            st.warning(f"Lỗi tile {i}: {ex}")
                        done += 1
                        progress.progress(done / len(bboxes))
                        status.text(f"Đã xong {done}/{len(bboxes)}")

            G = compose_graphs(graphs)
            if G is None:
                st.error("Không tải được ô nào.")
                st.stop()

            # NEW: PROJECT GRAPH
            Gp = ox.projection.project_graph(G)

            stats = compute_stats(Gp)
            st.success(f"Tổng KMU: **{stats['street_length_total_km']:.3f} km**")

            df = pd.DataFrame(rows).sort_values("tile_id")
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ CSV", df.to_csv(index=False).encode(), "tile_stats.csv")

            fig, ax = plot_graph(Gp, show=False, close=False, node_size=0, edge_linewidth=0.8)
            if show_tiles_outline:
                for row in rows:
                    n, s, e, w = row["north"], row["south"], row["east"], row["west"]
                    xs = [w, e, e, w, w]
                    ys = [s, s, n, n, s]
                    ax.plot(xs, ys, "r-", linewidth=0.8)

            st.pyplot(fig, clear_figure=True)

    except Exception as e:
        st.error("Có lỗi khi xử lý dữ liệu.")
        st.exception(e)
    finally:
        st.session_state["busy"] = False

else:
    st.info("Nhập thông tin → bấm **Tải & Tính toán**.")
