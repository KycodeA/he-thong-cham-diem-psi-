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

# Cấu hình trang giao diện rộng rãi, hiện đại
st.set_page_config(layout="wide", page_title="Hệ thống chấm điểm PSI mặt đường")

st.title("Ứng dụng chấm điểm Chỉ số PSI mặt đường - Phiên bản Tự động GPS")
st.subheader("Tích hợp AI quét dữ liệu hỗn hợp & Định vị GPS thực tế thiết bị")
st.markdown("---")

# ----------------------------------------------------------------
# TRẠNG THÁI HỆ THỐNG & ĐỊNH VỊ TỰ ĐỘNG
# ----------------------------------------------------------------
if 'hanh_trinh_dang_chay' not in st.session_state: st.session_state.hanh_trinh_dang_chay = False
if 'toa_do_bat_dau' not in st.session_state: st.session_state.toa_do_bat_dau = None
if 'toa_do_ket_thuc' not in st.session_state: st.session_state.toa_do_ket_thuc = None
if 'lich_su_lo_trinh' not in st.session_state: st.session_state.lich_su_lo_trinh = []
if 'thoi_gian_ket_thuc' not in st.session_state: st.session_state.thoi_gian_ket_thuc = ""
if 'ai_c_param' not in st.session_state: st.session_state.ai_c_param = 0.0
if 'ai_p_param' not in st.session_state: st.session_state.ai_p_param = 0.0
if 'ai_o_ga_count' not in st.session_state: st.session_state.ai_o_ga_count = 0

# Tọa độ mặc định khi chưa bật định vị (Hà Nội)
DEFAULT_LAT = 21.0274
DEFAULT_LON = 105.8046
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
# 🛰️ ĐOẠN CODE JAVASCRIPT ĐỊNH VỊ GPS THỰC TẾ QUA TRÌNH DUYỆT
# ----------------------------------------------------------------
st.sidebar.markdown("### 🛰️ Quản lý Hành trình GPS")

# Nhận dữ liệu tọa độ từ JavaScript gửi về Streamlit ngầm
gps_data = components.html(
    """
    <script>
    function sendLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const data = {
                        lat: position.coords.latitude,
                        lon: position.coords.longitude,
                        timestamp: new Date().getTime()
                    };
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: data
                    }, '*');
                },
                (error) => { console.error("Lỗi lấy GPS: ", error); },
                { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
            );
        }
    }
    // Tự động gọi lấy vị trí mỗi 4 giây để cập nhật lộ trình liên tục khi xe chạy
    setInterval(sendLocation, 4000);
    sendLocation(); // Chạy ngay lần đầu
    </script>
    """,
    height=0,
)

# Khởi tạo giá trị tọa độ hiện tại
current_lat = DEFAULT_LAT
current_lon = DEFAULT_LON

# Nếu nhận được tọa độ định vị thực tế từ thiết bị, ghi đè vào hệ thống
if gps_data is not None and isinstance(gps_data, dict) and 'lat' in gps_data:
    current_lat = gps_data['lat']
    current_lon = gps_data['lon']
    st.sidebar.success(f"📍 GPS Thực tế: {current_lat:.6f}, {current_lon:.6f}")
else:
    st.sidebar.warning("📡 Đang kết nối vệ tinh GPS... (Hoặc vui lòng bấm 'Cho phép chia sẻ vị trí')")

# Hiển thị số liệu tọa độ để sinh viên theo dõi (Đã mở khóa disabled=False để cho phép nhập tay khi mất mạng)
current_lat = st.sidebar.number_input("Vĩ độ hiện tại (Lat):", value=current_lat, format="%.6f", disabled=False)
current_lon = st.sidebar.number_input("Kinh độ hiện tại (Lon):", value=current_lon, format="%.6f", disabled=False)

# Tự động vẽ tuyến đường khi xe di chuyển nếu hành trình đang chạy
if st.session_state.hanh_trinh_dang_chay:
    current_point = (current_lat, current_lon)
    if not st.session_state.lich_su_lo_trinh or st.session_state.lich_su_lo_trinh[-1] != current_point:
        st.session_state.lich_su_lo_trinh.append(current_point)

# Điều khiển hành trình bằng nút bấm
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("🟢 Bắt đầu", use_container_width=True, disabled=st.session_state.hanh_trinh_dang_chay):
        st.session_state.hanh_trinh_dang_chay = True
        st.session_state.toa_do_bat_dau = (current_lat, current_lon)
        st.session_state.toa_do_ket_thuc = None
        st.session_state.lich_su_lo_trinh = [(current_lat, current_lon)]
with col_btn2:
    if st.button("🔴 Kết thúc", use_container_width=True, disabled=not st.session_state.hanh_trinh_dang_chay):
        st.session_state.hanh_trinh_dang_chay = False
        st.session_state.toa_do_ket_thuc = (current_lat, current_lon)
        st.session_state.thoi_gian_ket_thuc = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

# Chia giao diện chính
col1, col2 = st.columns([1, 1.2])

# ==========================================
# CỘT TRÁI: AI & THÔNG SỐ
# ==========================================
with col1:
    st.header("🧠 Phân tích mặt đường bằng AI (YOLOv8)")
    uploaded_files = st.file_uploader(
        "Tải lên đồng thời nhiều Ảnh và Video khảo sát hiện trường:", 
        type=["png", "jpg", "jpeg", "mp4"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        image_files = [f for f in uploaded_files if f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        video_files = [f for f in uploaded_files if f.name.lower().endswith('.mp4')]
        total_c, total_p, total_o_ga = 0.0, 0.0, 0
        samples_count = 0
        
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
                        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st_frame.image(img_rgb, caption=f"AI đang quét video...", use_container_width=True)
                cap.release()
                samples_count += (frame_count // 15 + 1)
        
        if image_files:
            st.info(f"📸 Đang quét bổ sung {len(image_files)} ảnh hiện trường...")
            cols_img = st.columns(min(len(image_files), 3))
            for idx, file in enumerate(image_files):
                img = Image.open(file)
                img_np = np.array(img)
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
            st.success("🎉 AI đã tổng hợp xong dữ liệu hỗn hợp!")

    st.write("---")
    st.header("📋 Thông số hình học & Kết cấu")
    ten_sv = st.text_input("Tên Sinh viên chấm điểm:", placeholder="Ví dụ: Nguyễn Văn A")
    ma_doan = st.text_input("Mã định danh đoạn đường:", placeholder="Ví dụ: 1_BTXM_1")
    loai_duong = st.selectbox("Chọn loại kết cấu mặt đường:", options=["Đường nhựa (Mặt đường mềm)", "Đường BTXM (Mặt đường cứng)"])
    
    c_param = st.number_input("% Diện tích nứt (C):", min_value=0.0, max_value=100.0, value=
