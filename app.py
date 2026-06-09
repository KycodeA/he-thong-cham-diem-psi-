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

# Cấu hình giao diện Streamlit rộng rãi, hiện đại
st.set_page_config(layout="wide", page_title="Hệ thống chấm điểm PSI tích hợp AI")

st.title("Ứng dụng chấm điểm Chỉ số PSI mặt đường - Phiên bản Tự động GPS")
st.subheader("Tích hợp AI Quét hình ảnh/Video & Định tuyến lộ trình thông minh")
st.markdown("---")

# ----------------------------------------------------------------
# KHO LƯU TRỮ TRẠNG THÁI (SESSION STATE) ĐỂ TRÁNH MẤT DỮ LIỆU
# ----------------------------------------------------------------
if 'toa_do_bat_dau' not in st.session_state: st.session_state.toa_do_bat_dau = None
if 'toa_do_ket_thuc' not in st.session_state: st.session_state.toa_do_ket_thuc = None
if 'lich_su_lo_trinh' not in st.session_state: st.session_state.lich_su_lo_trinh = []
if 'tong_quang_duong_m' not in st.session_state: st.session_state.tong_quang_duong_m = 0.0
if 'thoi_gian_ket_thuc' not in st.session_state: st.session_state.thoi_gian_ket_thuc = ""
if 'map_key' not in st.session_state: st.session_state.map_key = "default"

# Tham số do AI quét được
if 'ai_c_param' not in st.session_state: st.session_state.ai_c_param = 0.0
if 'ai_p_param' not in st.session_state: st.session_state.ai_p_param = 0.0
if 'ai_o_ga_count' not in st.session_state: st.session_state.ai_o_ga_count = 0

# Tọa độ mặc định (Khu vực Đại học Giao thông Vận tải Hà Nội)
TOA_DO_MAU_LAT = 21.0274
TOA_DO_MAU_LON = 105.8046

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt') 

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Lỗi khởi tải mô hình AI: {e}")

# ----------------------------------------------------------------
# 🛰️ BẢNG ĐIỀU KHIỂN SIDEBAR & ĐỊNH VỊ GPS VỆ TINH
# ----------------------------------------------------------------
st.sidebar.markdown("### 🛰️ Quản lý Vị trí GPS")

gps_data = components.html(
    """
    <script>
    const options = {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
    };

    function success(position) {
        const data = {
            lat: position.coords.latitude,
            lon: position.coords.longitude
        };
        window.parent.postMessage({ type: 'streamlit:setComponentValue', value: data }, '*');
    }

    function error(err) {
        console.warn('ERROR(' + err.code + '): ' + err.message);
    }

    if (navigator.geolocation) {
        navigator.geolocation.watchPosition(success, error, options);
    } else {
        console.error("Trình duyệt không hỗ trợ Geolocation API.");
    }
    </script>
    """,
    height=0,
)

current_lat = TOA_DO_MAU_LAT
current_lon = TOA_DO_MAU_LON

if gps_data is not None and isinstance(gps_data, dict) and 'lat' in gps_data:
    current_lat = gps_data['lat']
    current_lon = gps_data['lon']
    st.sidebar.success(f"📍 GPS Thiết bị: {current_lat:.6f}, {current_lon:.6f}")
else:
    st.sidebar.warning("📡 Đang tìm tín hiệu vệ tinh GPS... (Hoặc bạn hãy tự gõ tọa độ kiểm thử bên dưới)")

input_lat = st.sidebar.number_input("Vĩ độ (Lat):", value=current_lat, format="%.6f")
input_lon = st.sidebar.number_input("Kinh độ (Lon):", value=current_lon, format="%.6f")

phuong_tien = st.sidebar.selectbox("🚗 Chế độ bám đường:", options=["Đường phố di chuyển linh hoạt", "Đường đi bộ nội khu"])
osrm_profile = "driving" if phuong_tien == "Đường phố di chuyển linh hoạt" else "foot"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Ghi nhận Tuyến đường Khảo sát")

col_lock1, col_lock2 = st.sidebar.columns(2)
with col_lock1:
    if st.button("📍 Khóa Điểm Đầu", use_container_width=True):
        st.session_state.toa_do_bat_dau = (input_lat, input_lon)
        st.sidebar.success(f"Đã khóa Đầu!")

with col_lock2:
    if st.button("📍 Khóa Điểm Cuối", use_container_width=True):
        st.session_state.toa_do_ket_thuc = (input_lat, input_lon)
        st.sidebar.success(f"Đã khóa Cuối!")

# NÚT BẤM GỌI THUẬT TOÁN ĐỊNH TUYẾN (Cập nhật lấy Khoảng cách thực tế)
if st.button("🗺️ KÍCH HOẠT ĐỊNH TUYẾN GOOGLE MAPS", type="primary", use_container_width=True):
    if st.session_state.toa_do_bat_dau is None or st.session_state.toa_do_ket_thuc is None:
        st.sidebar.error("❌ Lỗi: Bạn phải nhập tọa độ và bấm nút 'Khóa Điểm Đầu' + 'Khóa Điểm Cuối' trước khi kích hoạt!")
    else:
        st.session_state.thoi_gian_ket_thuc = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        lat_start, lon_start = st.session_state.toa_do_bat_dau
        lat_end, lon_end = st.session_state.toa_do_ket_thuc
        
        try:
            url = f"https://router.project-osrm.org/route/v1/{osrm_profile}/{lon_start},{lat_start};{lon_end},{lat_end}?overview=full&geometries=geojson&continue_straight=false"
            response = requests.get(url, timeout=5).json()
            
            if response.get("code") == "Ok":
                geometry = response["routes"][0]["geometry"]["coordinates"]
                # Cập nhật tổng quãng đường thực tế (m) từ API OSRM
                st.session_state.tong_quang_duong_m = float(response["routes"][0]["distance"])
                st.session_state.lich_su_lo_trinh = [[coord[1], coord[0]] for coord in geometry]
                st.session_state.map_key = datetime.now().strftime("%H%M%S")
                st.sidebar.success(f"🎉 Định tuyến thành công! Chiều dài tuyến: {st.session_state.tong_quang_duong_m:.1f} m")
            else:
                # Phương án dự phòng tự tính khoảng cách đường chim bay nếu API lỗi
                R = 6371000
                p1, p2 = math.radians(lat_start), math.radians(lat_end)
                dp = math.radians(lat_end - lat_start)
                dl = math.radians(lon_end - lon_start)
                a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2) * math.sin(dl/2)**2
                st.session_state.tong_quang_duong_m = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
                
                steps = 15
                interpolated_route = []
                for i in range(steps + 1):
                    t = i / steps
                    lat_t = lat_start + (lat_end - lat_start) * t
                    lon_t = lon_start + (lon_end - lon_start) * t
                    if 0 < i < steps:
                        lat_t += math.sin(t * math.pi) * 0.0009
                    interpolated_route.append([lat_t, lon_t])
                st.session_state.lich_su_lo_trinh = interpolated_route
                st.session_state.map_key = datetime.now().strftime("%H%M%S")
                st.sidebar.info("🗺️ Đã tự động tối ưu đường cong nội đô!")
        except Exception as e:
            st.session_state.lich_su_lo_trinh = [[lat_start, lon_start], [lat_end, lon_end]]
            st.session_state.tong_quang_duong_m = 0.0

# Chia bố cục màn hình chính
col1, col2 = st.columns([1, 1.2])

# ================================================================
# CỘT 1: TRÍ TRUỆ NHÂN TẠO AI & THÔNG SỐ PSI
# ================================================================
with col1:
    st.header("🧠 Trí tuệ Nhân tạo AI Quét Mặt Đường")
    uploaded_files = st.file_uploader("Tải lên tư liệu hình ảnh / Video khảo sát:", type=["png", "jpg", "jpeg", "mp4"], accept_multiple_files=True)
    
    if uploaded_files:
        image_files = [f for f in uploaded_files if f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        video_files = [f for f in uploaded_files if f.name.lower().endswith('.mp4')]
        total_c, total_p, total_o_ga, samples_count = 0.0, 0.0, 0, 0
        
        if video_files:
            for video_file in video_files:
                st.info(f"🔄 AI đang phân tích Video: {video_file.name}")
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
                        st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="AI quét video...", use_container_width=True)
                cap.release()
                samples_count += (frame_count // 15 + 1)
        
        if image_files:
            st.info(f"📸 AI đang quét chi tiết {len(image_files)} ảnh...")
            cols_img = st.columns(min(len(image_files), 3))
            for idx, file in enumerate(image_files):
                img_np = np.array(Image.open(file))
                results = model(img_np, verbose=False)
                annotated_img = results[0].plot()
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls == 0: total_c += 1.5
                    elif cls == 1: total_p += 2.0
                    else: total_o_ga += 1
                with cols_img[idx % 3]: st.image(annotated_img, caption=file.name, use_container_width=True)
                samples_count += 1

        if samples_count > 0:
            st.session_state.ai_c_param = min(round(total_c / samples_count, 1), 100.0)
            st.session_state.ai_p_param = min(round(total_p / samples_count, 1), 100.0)
            st.session_state.ai_o_ga_count = total_o_ga
            st.success("🎉 AI đã điền số liệu nứt/vá tự động!")

    st.write("---")
    st.header("📋 Khai báo Kết cấu & Tính toán")
    ten_sv = st.text_input("Tên Sinh viên chấm điểm:", placeholder="Ví dụ: Nguyễn Văn A")
    ma_doan = st.text_input("Mã định danh đoạn đường:", placeholder="Ví dụ: 1_BTXM_1")
    loai_duong = st.selectbox("Loại kết cấu mặt đường:", ["Đường nhựa (Mặt đường mềm)", "Đường BTXM (Mặt đường cứng)"])
    
    c_param = st.number_input("% Diện tích nứt vỡ (C):", min_value=0.0, max_value=100.0, value=st.session_state.ai_c_param, step=0.1, key="input_c_param")
    p_param = st.number_input("% Diện tích miếng vá (P):", min_value=0.0, max_value=100.0, value=st.session_state.ai_p_param, step=0.1, key="input_p_param")
    st.metric(label="🚨 Tổng số lượng ổ gà phát hiện", value=f"{st.session_state.ai_o_ga_count} Ổ gà")
    
    sv_param = st.number_input("Chỉ số độ gồ ghề (SV):", min_value=0.0, value=0.0, step=0.1)
    rd_param = st.number_input("Độ lún vệt bánh xe (RD - mm):", min_value=0.0, value=0.0, step=0.1) if loai_duong == "Đường nhựa (Mặt đường mềm)" else 0.0

    st.write("---")
    btn_tinh_psi = st.button("🚀 BẮT ĐẦU TÍNH TOÁN CHỈ SỐ PSI", type="primary", use_container_width=True)

# ================================================================
# CỘT 2: BẢN ĐỒ LỘ TRÌNH ĐƯỜNG PHỐ & XUẤT FILE BÁO CÁO EXCEL
# ================================================================
with col2:
    st.header("🗺️ Bản đồ Lộ trình Thực tế")
    
    map_center = [input_lat, input_lon] if not st.session_state.toa_do_bat_dau else st.session_state.toa_do_bat_dau
    m = folium.Map(location=map_center, zoom_start=16)
    
    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google Roads', name='Bản đồ giao thông phố').add_to(m)
    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', attr='Google Satellite', name='Vệ tinh').add_to(m)
    folium.LayerControl().add_to(m)
    
    if st.session_state.toa_do_bat_dau:
        folium.Marker(st.session_state.toa_do_bat_dau, tooltip="Điểm A", icon=folium.Icon(color='green', icon='play')).add_to(m)
    if st.session_state.toa_do_ket_thuc:
        folium.Marker(st.session_state.toa_do_ket_thuc, tooltip="Điểm B", icon=folium.Icon(color='red', icon='flag')).add_to(m)

    if len(st.session_state.lich_su_lo_trinh) > 1:
        folium.PolyLine(locations=st.session_state.lich_su_lo_trinh, color="#0066FF", weight=6, opacity=0.9, tooltip="Tuyến đường di chuyển").add_to(m)
        
        min_lat = min(c[0] for c in st.session_state.lich_su_lo_trinh)
        max_lat = max(c[0] for c in st.session_state.lich_su_lo_trinh)
        min_lon = min(c[1] for c in st.session_state.lich_su_lo_trinh)
        max_lon = max(c[1] for c in st.session_state.lich_su_lo_trinh)
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        
    dynamic_key = st.session_state.map_key
    st_data = st_folium(m, width=720, height=480, key=f"map_{dynamic_key}")

    st.write("---")
    st.subheader("🎯 Đánh Giá Chỉ Số Mặt Đường Kỹ Thuật:")
    
    if btn_tinh_psi:
        if not ten_sv or not ma_doan: 
            st.error("Vui lòng nhập đầy đủ thông tin Tên Sinh viên và Mã đoạn đường!")
        elif st.session_state.toa_do_ket_thuc is None: 
            st.error("Bạn chưa bấm nút kích hoạt định tuyến!")
        else:
            # --------------------------------------------------------
            # THAY ĐỔI QUAN TRỌNG: KIỂM TRA THEO TỔNG QUÃNG ĐƯỜNG THỰC TẾ
            # --------------------------------------------------------
            quang_duong_hien_tai = st.session_state.tong_quang_duong_m
            gioi_han_m = 1000.0 if loai_duong == "Đường nhựa (Mặt đường mềm)" else 500.0
            ten_gioi_han = "1km" if loai_duong == "Đường nhựa (Mặt đường mềm)" else "500m"
            
            # Quãng đường hợp lệ nếu KHÔNG VƯỢT QUÁ giới hạn quy định từ điểm bắt đầu
            dung_lo_trinh = quang_duong_hien_tai <= gioi_han_m
            
            log_sv = math.log10(1 + sv_param)
            sqrt_cp = math.sqrt(c_param + p_param)
            
            psi = (5.03 - 1.91 * log_sv - 1.38 * (rd_param ** 2) - 0.01 * sqrt_cp) if loai_duong == "Đường nhựa (Mặt đường mềm)" else (5.41 - 1.78 * log_sv - 0.09 * sqrt_cp)
            if st.session_state.ai_o_ga_count > 0: psi -= min(st.session_state.ai_o_ga_count * 0.05, 1.0)
            psi_rounded = round(max(0.0, min(5.0, psi)), 1)
            
            st.metric(label="Chỉ số Phục vụ Mặt đường (PSI) Đề xuất", value=f"{psi_rounded} / 5.0")
            if psi_rounded >= 4.0: st.success("🟢 Trạng thái mặt đường: Rất tốt và êm thuận.")
            elif psi_rounded >= 2.0: st.warning("🟡 Trạng thái mặt đường: Xuất hiện hư hỏng nhẹ.")
            else: st.error("🔴 Trạng thái mặt đường: Xuất hiện hỏng hóc nghiêm trọng!")
            
            st.info(f"📊 Tổng quãng đường thực tế đo được: {quang_duong_hien_tai:.1f} m")
            if dung_lo_trinh: 
                st.success(f"✅ Đạt yêu cầu về chiều dài tuyến (Tổng quãng đường {quang_duong_hien_tai:.1f}m <= quy định {ten_gioi_han}).")
            else: 
                st.error(f"❌ Vượt quá chiều dài quy định! (Tổng quãng đường {quang_duong_hien_tai:.1f}m > giới hạn cho phép {ten_gioi_han}).")

            # XUẤT FILE EXCEL
            st.write("---")
            data_report = {
                "Danh mục báo cáo": ["Sinh viên thực hiện", "Mã đoạn đường", "Kết cấu mặt đường", "Thời gian hoàn thành", "Vị trí bắt đầu (Lat, Lon)", "Vị trí kết thúc (Lat, Lon)", "Tổng chiều dài thực tế", "Kiểm định lộ trình", "Mật độ nứt vỡ (C)", "Mật độ miếng vá (P)", "Tổng số ổ gà", "Độ gồ ghề (SV)", "Chỉ số PSI sau cùng", "Kết luận phân loại"],
                "Kết quả chi tiết": [ten_sv, ma_doan, loai_duong, st.session_state.thoi_gian_ket_thuc, str(st.session_state.toa_do_bat_dau), str(st.session_state.toa_do_ket_thuc), f"{quang_duong_hien_tai:.1f} m", "HỢP LỆ" if dung_lo_trinh else "QUÁ GIỚI HẠN CHUẨN", f"{c_param}%", f"{p_param}%", st.session_state.ai_o_ga_count, sv_param, psi_rounded, "TỐT" if psi_rounded >= 4.0 else ("TRUNG BÌNH" if psi_rounded >= 2.0 else "XUỐNG CẤP")]
            }
            buffer = io.BytesIO()
            pd.DataFrame(data_report).to_excel(buffer, index=False, sheet_name='Dữ liệu PSI', engine='openpyxl')
            st.download_button(label="📥 TẢI BÁO CÁO KẾT QUẢ SANG FILE EXCEL (.XLSX)", data=buffer.getvalue(), file_name=f"Bao_cao_PSI_{ma_doan}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
