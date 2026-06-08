import streamlit as st
import math
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import io
import requests

# Cấu hình trang giao diện rộng rãi, hiện đại
st.set_page_config(layout="wide", page_title="Hệ thống chấm điểm PSI mặt đường")

st.title("Ứng dụng chấm điểm Chỉ số PSI mặt đường - Phiên bản Tự động GPS")
st.subheader("Tích hợp AI quét dữ liệu hỗn hợp & Định tuyến lộ trình thông minh")
st.markdown("---")

# ----------------------------------------------------------------
# TRẠNG THÁI HỆ THỐNG & ĐỊNH VỊ TỰ ĐỘNG
# ----------------------------------------------------------------
if 'hanh_trinh_dang_chay' not in st.session_state: st.session_state.hanh_trinh_dang_chay = False
if 'toa_do_bat_dau' not in st.session_state: st.session_state.toa_do_bat_dau = None
if 'toa_do_ket_thuc' not in st.session_state: st.session_state.toa_do_ket_thuc = None
if 'lich_su_lo_trinh' not in st.session_state: st.session_state.lich_su_lo_trinh = []
if 'thoi_gian_ket_thuc' not in st.session_state: st.session_state.thoi_gian_ket_thuc = ""
if 'loi_ban_do' not in st.session_state: st.session_state.loi_ban_do = False
if 'ai_c_param' not in st.session_state: st.session_state.ai_c_param = 0.0
if 'ai_p_param' not in st.session_state: st.session_state.ai_p_param = 0.0
if 'ai_o_ga_count' not in st.session_state: st.session_state.ai_o_ga_count = 0

# Tọa độ gốc khu vực mẫu (Trường UTC)
TOA_DO_MAU_LAT = 21.0274
TOA_DO_MAU_LON = 105.8046

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt') 

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Lỗi tải mô hình AI: {e}")

# ----------------------------------------------------------------
# 🛰️ BẢNG ĐIỀU KHIỂN & ĐỊNH VỊ GPS BÊN SIDEBAR
# ----------------------------------------------------------------
st.sidebar.markdown("### 🛰️ Quản lý Hành trình GPS")

# Lấy GPS ngầm từ thiết bị
gps_data = components.html(
    """
    <script>
    function sendLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const data = {
                        lat: position.coords.latitude,
                        lon: position.coords.longitude
                    };
                    window.parent.postMessage({ type: 'streamlit:setComponentValue', value: data }, '*');
                },
                (error) => { console.error("Lỗi lấy GPS: ", error); },
                { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
            );
        }
    }
    setInterval(sendLocation, 4000);
    sendLocation();
    </script>
    """,
    height=0,
)

current_lat = TOA_DO_MAU_LAT
current_lon = TOA_DO_MAU_LON

if gps_data is not None and isinstance(gps_data, dict) and 'lat' in gps_data:
    current_lat = gps_data['lat']
    current_lon = gps_data['lon']
    st.sidebar.success(f"📍 GPS Thực tế: {current_lat:.6f}, {current_lon:.6f}")
else:
    st.sidebar.warning("📡 Đang kết nối vệ tinh GPS... (Hoặc nhập tọa độ thủ công)")

# Khung nhập liệu tọa độ
input_lat = st.sidebar.number_input("Vĩ độ (Lat):", value=current_lat, format="%.6f")
input_lon = st.sidebar.number_input("Kinh độ (Lon):", value=current_lon, format="%.6f")

# Chọn phương tiện để gọi API định tuyến
phuong_tien = st.sidebar.selectbox("🚗 Chọn phương thức tìm đường:", options=["Đi bộ (Walk)", "Xe máy (Bike)", "Ô tô (Car)"])
osrm_profile = "foot" if phuong_tien == "Đi bộ (Walk)" else ("bicycle" if phuong_tien == "Xe máy (Bike)" else "driving")

# Theo dõi trực tuyến nếu xe đang chạy (Chưa chốt tuyến)
if st.session_state.hanh_trinh_dang_chay:
    current_point = (input_lat, input_lon)
    if not st.session_state.lich_su_lo_trinh or st.session_state.lich_su_lo_trinh[-1] != current_point:
        st.session_state.lich_su_lo_trinh.append(current_point)

# Nút điều khiển
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🟢 Bắt đầu", use_container_width=True):
        st.session_state.hanh_trinh_dang_chay = True
        st.session_state.loi_ban_do = False
        st.session_state.toa_do_bat_dau = (input_lat, input_lon)
        st.session_state.toa_do_ket_thuc = None
        st.session_state.lich_su_lo_trinh = [(input_lat, input_lon)]
        st.sidebar.info("📍 Đã ghi nhận Điểm Bắt Đầu.")

with col_btn2:
    if st.button("🔴 Kết thúc", use_container_width=True):
        st.session_state.hanh_trinh_dang_chay = False
        st.session_state.toa_do_ket_thuc = (input_lat, input_lon)
        st.session_state.thoi_gian_ket_thuc = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        
        lat_start, lon_start = st.session_state.toa_do_bat_dau
        lat_end, lon_end = st.session_state.toa_do_ket_thuc
        
        # --- GỌI API ĐỊNH TUYẾN THỰC TẾ OSRM KHI KẾT THÚC ---
        try:
            # OSRM URL chuẩn bảo mật https, đổi vị trí {lon},{lat}
            url = f"https://router.project-osrm.org/route/v1/{osrm_profile}/{lon_start},{lat_start};{lon_end},{lat_end}?overview=full&geometries=geojson"
            response = requests.get(url, timeout=5).json()
            
            if response.get("code") == "Ok":
                geometry = response["routes"][0]["geometry"]["coordinates"]
                st.session_state.lich_su_lo_trinh = [[coord[1], coord[0]] for coord in geometry]
                st.session_state.loi_ban_do = False
                st.session_state.map_key = datetime.now().strftime("%H%M%S")
                st.sidebar.success(f"🗺️ Đã bám sát lộ trình [{phuong_tien}]!")
            else:
                st.session_state.loi_ban_do = True
                st.sidebar.error("Lỗi: Khu vực không có đường đi cho phương tiện này.")
                st.session_state.lich_su_lo_trinh = [[lat_start, lon_start], [lat_end, lon_end]]
        except Exception as e:
            st.session_state.loi_ban_do = True
            st.sidebar.warning(f"Mất kết nối API Bản đồ. Chuyển về nét đứt.")
            st.session_state.lich_su_lo_trinh = [[lat_start, lon_start], [lat_end, lon_end]]

# Chia giao diện chính
col1, col2 = st.columns([1, 1.2])

# ==========================================
# CỘT TRÁI: AI & TÍNH TOÁN PSI
# ==========================================
with col1:
    st.header("🧠 Phân tích mặt đường AI (YOLOv8)")
    uploaded_files = st.file_uploader(
        "Tải lên Ảnh / Video khảo sát hiện trường:", 
        type=["png", "jpg", "jpeg", "mp4"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        image_files = [f for f in uploaded_files if f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        video_files = [f for f in uploaded_files if f.name.lower().endswith('.mp4')]
        total_c, total_p, total_o_ga, samples_count = 0.0, 0.0, 0, 0
        
        if video_files:
            for video_file in video_files:
                st.info(f"🔄 Đang quét Video: {video_file.name}")
                with open("temp_video.mp4", "wb") as f: f.write(video_file.read())
                cap = cv2.VideoCapture("temp_video.mp4")
                st_frame = st.empty()
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frame_count += 1
                    if frame_count % 15 == 0:
                        results = model(frame, verbose=False)
                        annotated_frame = results[0].plot()
                        for box in results[0].boxes:
                            cls = int(box.cls[0])
                            if cls == 0: total_c += 0.5
                            elif cls == 1: total_p += 0.8
                            else: total_o_ga += 1
                        st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"AI quét video...", use_container_width=True)
                cap.release()
                samples_count += (frame_count // 15 + 1)
        
        if image_files:
            st.info(f"📸 Đang quét {len(image_files)} ảnh hiện trường...")
            cols_img = st.columns(min(len(image_files), 3))
            for idx, file in enumerate(image_files):
                img_np = np.array(Image.open(file))
                results = model(img_np, verbose=False)
                annotated_img = results[0].plot()
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls == 0: total_c += 1.2
                    elif cls == 1: total_p += 1.5
                    else: total_o_ga += 1
                with cols_img[idx % 3]:
                    st.image(annotated_img, caption=file.name, use_container_width=True)
                samples_count += 1

        if samples_count > 0:
            st.session_state.ai_c_param = min(round(total_c / samples_count, 1), 100.0)
            st.session_state.ai_p_param = min(round(total_p / samples_count, 1), 100.0)
            st.session_state.ai_o_ga_count = total_o_ga
            st.success("🎉 AI đã tổng hợp xong dữ liệu!")

    st.write("---")
    st.header("📋 Thông số & Kết cấu")
    ten_sv = st.text_input("Tên Sinh viên:", placeholder="Nguyễn Văn A")
    ma_doan = st.text_input("Mã đoạn đường:", placeholder="1_BTXM_1")
    loai_duong = st.selectbox("Loại kết cấu:", ["Đường nhựa (Mặt đường mềm)", "Đường BTXM (Mặt đường cứng)"])
    
    c_param = st.number_input("% Diện tích nứt (C):", min_value=0.0, max_value=100.0, value=st.session_state.ai_c_param, step=0.1)
    p_param = st.number_input("% Diện tích vá (P):", min_value=0.0, max_value=100.0, value=st.session_state.ai_p_param, step=0.1)
    st.metric(label="🚨 Tổng số ổ gà", value=f"{st.session_state.ai_o_ga_count} Ổ gà")
    sv_param = st.number_input("Độ gồ ghề (SV):", min_value=0.0, value=0.0, step=0.1)
    rd_param = st.number_input("Lún vệt bánh xe (RD - mm):", min_value=0.0, value=0.0, step=0.1) if loai_duong == "Đường nhựa (Mặt đường mềm)" else 0.0

    st.write("---")
    btn_tinh_psi = st.button("Tính điểm PSI", type="primary")

# ==========================================
# CỘT PHẢI: BẢN ĐỒ VỆ TINH & BÁO CÁO
# ==========================================
with col2:
    st.header("🗺️ Bản đồ Lộ trình Thông minh")
    
    # Khởi tạo bản đồ cơ bản
    map_center = [input_lat, input_lon] if not st.session_state.toa_do_bat_dau else st.session_state.toa_do_bat_dau
    m = folium.Map(location=map_center, zoom_start=17)
    
    # Thêm lớp Vệ tinh hiển thị rõ nét
    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Vệ Tinh', name='Vệ tinh', overlay=False).add_to(m)
    folium.TileLayer('openstreetmap', name='Đường phố').add_to(m)
    folium.LayerControl().add_to(m)
    
    # Đánh dấu Marker
    folium.Marker([input_lat, input_lon], popup="Vị trí hiện tại", icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
    if st.session_state.toa_do_bat_dau:
        folium.Marker(st.session_state.toa_do_bat_dau, tooltip="Điểm Bắt Đầu", icon=folium.Icon(color='green', icon='play')).add_to(m)
    if st.session_state.toa_do_ket_thuc:
        folium.Marker(st.session_state.toa_do_ket_thuc, tooltip="Điểm Kết Thúc", icon=folium.Icon(color='red', icon='flag')).add_to(m)

    # Vẽ tuyến đường & Auto-Zoom
    if len(st.session_state.lich_su_lo_trinh) > 1:
        if st.session_state.loi_ban_do:
            # Lỗi API -> Vẽ đường chim bay đứt nét
            folium.PolyLine(locations=st.session_state.lich_su_lo_trinh, color="gray", weight=4, dash_array="10, 10", tooltip="Lỗi định tuyến (Đường chim bay)").add_to(m)
        else:
            # Thành công API -> Vẽ đường uốn lượn thực tế
            folium.PolyLine(locations=st.session_state.lich_su_lo_trinh, color="#00E5FF", weight=6, opacity=0.9, tooltip=f"Lộ trình {phuong_tien}").add_to(m)
        
        # --- THUẬT TOÁN AUTO-FIT BOUNDS (TỰ ĐỘNG PHÓNG TO BAO TRỌN LỘ TRÌNH) ---
        min_lat = min(c[0] for c in st.session_state.lich_su_lo_trinh)
        max_lat = max(c[0] for c in st.session_state.lich_su_lo_trinh)
        min_lon = min(c[1] for c in st.session_state.lich_su_lo_trinh)
        max_lon = max(c[1] for c in st.session_state.lich_su_lo_trinh)
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        
    dynamic_key = st.session_state.get("map_key", "default_map")
    st_data = st_folium(m, width=700, height=450, key=f"map_{dynamic_key}")

    st.write("---")
    st.subheader("🎯 Đánh giá Hệ thống:")
    
    if btn_tinh_psi:
        if not ten_sv or not ma_doan:
            st.error("Vui lòng điền Tên SV và Mã đoạn đường!")
        elif not st.session_state.toa_do_ket_thuc:
            st.error("Bạn chưa bấm Kết thúc hành trình!")
        else:
            lat1, lon1 = st.session_state.toa_do_ket_thuc
            R = 6371000
            phi1, phi2 = math.radians(lat1), math.radians(TOA_DO_MAU_LAT)
            dphi = math.radians(TOA_DO_MAU_LAT - lat1)
            dlam = math.radians(TOA_DO_MAU_LON - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlam/2)**2
            khoang_cach = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
            
            pham_vi, ten_pham_vi = (1000.0, "1km") if loai_duong == "Đường nhựa (Mặt đường mềm)" else (500.0, "500m")
            dung_lo_trinh = khoang_cach <= pham_vi
            
            log_sv = math.log10(1 + sv_param)
            sqrt_cp = math.sqrt(c_param + p_param)
            
            psi = (5.03 - 1.91 * log_sv - 1.38 * (rd_param ** 2) - 0.01 * sqrt_cp) if loai_duong == "Đường nhựa (Mặt đường mềm)" else (5.41 - 1.78 * log_sv - 0.09 * sqrt_cp)
            if st.session_state.ai_o_ga_count > 0: psi -= min(st.session_state.ai_o_ga_count * 0.05, 1.0)
            psi_rounded = round(max(0.0, min(5.0, psi)), 1)
            
            st.metric(label="Chỉ số PSI Mặt đường", value=f"{psi_rounded} / 5.0")
            if psi_rounded >= 4.0: st.success("🟢 Rất tốt.")
            elif psi_rounded >= 2.0: st.warning("🟡 Rung lắc nhẹ.")
            else: st.error("🔴 Nguy hiểm!")
            
            if dung_lo_trinh: st.success(f"✅ Đúng lộ trình (Trong phạm vi {ten_pham_vi}).")
            else: st.error(f"❌ Sai lộ trình (Lệch {khoang_cach:.1f}m, vượt mức {ten_pham_vi}).")

            # XUẤT EXCEL
            st.write("---")
            data_report = {
                "Thông số báo cáo": [
                    "Tên Sinh viên", "Mã đoạn đường", "Loại mặt đường", "Thời gian", "Tọa độ Bắt đầu", "Tọa độ Kết thúc", 
                    "Đúng lộ trình", "% Diện tích nứt (C)", "% Diện tích vá (P)", "Số ổ gà", "SV", "PSI", "Đánh giá"
                ],
                "Giá trị thực tế": [
                    ten_sv, ma_doan, loai_duong, st.session_state.thoi_gian_ket_thuc, 
                    str(st.session_state.toa_do_bat_dau), str(st.session_state.toa_do_ket_thuc),
                    "Đúng" if dung_lo_trinh else "Sai", c_param, p_param, st.session_state.ai_o_ga_count, 
                    sv_param, psi_rounded, "Tốt" if psi_rounded >= 4.0 else ("Trung bình" if psi_rounded >= 2.0 else "Hỏng")
                ]
            }
            buffer = io.BytesIO()
            pd.DataFrame(data_report).to_excel(buffer, index=False, sheet_name='Báo cáo PSI', engine='openpyxl')
            st.download_button(label="📥 Tải Báo cáo (.xlsx)", data=buffer.getvalue(), file_name=f"PSI_{ma_doan}_{ten_sv}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
