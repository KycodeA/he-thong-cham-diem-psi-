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

st.title("Ứng dụng chấm điểm Chỉ số PSI mặt đường - Phiên bản Hoàn thiện")
st.subheader("Tích hợp AI quét dữ liệu hỗn hợp & Xuất báo cáo kết quả")
st.markdown("---")

# ----------------------------------------------------------------
# TRẠNG THÁI HỆ THỐNG & ĐỊNH VỊ (Giữ nguyên cấu trúc Bước 4)
# ----------------------------------------------------------------
if 'hanh_trinh_dang_chay' not in st.session_state: st.session_state.hanh_trinh_dang_chay = False
if 'toa_do_bat_dau' not in st.session_state: st.session_state.toa_do_bat_dau = None
if 'toa_do_ket_thuc' not in st.session_state: st.session_state.toa_do_ket_thuc = None
if 'lich_su_lo_trinh' not in st.session_state: st.session_state.lich_su_lo_trinh = []
if 'thoi_gian_ket_thuc' not in st.session_state: st.session_state.thoi_gian_ket_thuc = ""
if 'ai_c_param' not in st.session_state: st.session_state.ai_c_param = 0.0
if 'ai_p_param' not in st.session_state: st.session_state.ai_p_param = 0.0
if 'ai_o_ga_count' not in st.session_state: st.session_state.ai_o_ga_count = 0

TOA_DO_MAU_LAT = 21.0274
TOA_DO_MAU_LON = 105.8046

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt') 

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Lỗi tải mô hình AI: {e}")

# --- THANH SIDEBAR QUẢN LÝ ---
st.sidebar.markdown("### 🛰️ Quản lý Hành trình GPS")
current_lat = st.sidebar.number_input("Vĩ độ hiện tại (Lat):", value=21.0274, format="%.6f")
current_lon = st.sidebar.number_input("Kinh độ hiện tại (Lon):", value=105.8046, format="%.6f")

if st.session_state.hanh_trinh_dang_chay:
    current_point = (current_lat, current_lon)
    if not st.session_state.lich_su_lo_trinh or st.session_state.lich_su_lo_trinh[-1] != current_point:
        st.session_state.lich_su_lo_trinh.append(current_point)

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

# Chia giao diện chính làm 2 cột độc lập
col1, col2 = st.columns([1, 1.2])

# ==========================================
# CỘT TRÁI: KHU VỰC QUÉT AI HỖN HỢP & BẢNG SỐ LIỆU
# ==========================================
with col1:
    st.header("🧠 Phân tích mặt đường bằng AI (YOLOv8)")
    
    # Cho phép tải lên cả Ảnh và Video cùng một lúc vào ô nhận dữ liệu
    uploaded_files = st.file_uploader(
        "Tải lên đồng thời nhiều Ảnh và Video khảo sát hiện trường:", 
        type=["png", "jpg", "jpeg", "mp4"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Phân loại file ngay khi người dùng upload hỗn hợp
        image_files = [f for f in uploaded_files if f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        video_files = [f for f in uploaded_files if f.name.lower().endswith('.mp4')]
        
        total_c, total_p, total_o_ga = 0.0, 0.0, 0
        samples_count = 0
        
        # 1. XỬ LÝ TOÀN BỘ VIDEO TRONG DANH SÁCH TẢI LÊN
        if video_files:
            for v_idx, video_file in enumerate(video_files):
                st.info(f"🔄 Đang quét Video: {video_file.name}")
                with open("temp_video.mp4", "wb") as f:
                    f.write(video_file.read())
                
                cap = cv2.VideoCapture("temp_video.mp4")
                st_frame = st.empty()
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frame_count += 1
                    
                    if frame_count % 15 == 0: # Cứ 15 khung hình lấy 1
                        results = model(frame, verbose=False)
                        annotated_frame = results[0].plot()
                        
                        for box in results[0].boxes:
                            cls = int(box.cls[0])
                            if cls == 0: total_c += 0.5
                            elif cls == 1: total_p += 0.8
                            else: total_o_ga += 1
                        
                        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st_frame.image(img_rgb, caption=f"AI đang quét video {video_file.name}...", use_container_width=True)
                
                cap.release()
                samples_count += (frame_count // 15 + 1)
        
        # 2. XỬ LÝ TOÀN BỘ ẢNH TRONG DANH SÁCH TẢI LÊN (Chạy song song)
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
                    st.image(annotated_img, caption=f"Kết quả {file.name}", use_container_width=True)
                samples_count += 1

        # Tổng hợp dữ liệu từ cả 2 nguồn Ảnh + Video
        if samples_count > 0:
            st.session_state.ai_c_param = min(round(total_c / samples_count, 1), 100.0)
            st.session_state.ai_p_param = min(round(total_p / samples_count, 1), 100.0)
            st.session_state.ai_o_ga_count = total_o_ga
            st.success("🎉 AI đã tổng hợp xong dữ liệu từ tất cả Ảnh và Video của bạn!")

    st.write("---")
    st.header("📋 Thông số hình học & Kết cấu")
    ten_sv = st.text_input("Tên Sinh viên chấm điểm:", placeholder="Ví dụ: Nguyễn Văn A")
    ma_doan = st.text_input("Mã định danh đoạn đường:", placeholder="Ví dụ: 1_BTXM_1")
    loai_duong = st.selectbox("Chọn loại kết cấu mặt đường:", options=["Đường nhựa (Mặt đường mềm)", "Đường BTXM (Mặt đường cứng)"])
    
    c_param = st.number_input("% Diện tích nứt (C):", min_value=0.0, max_value=100.0, value=st.session_state.ai_c_param, step=0.1)
    p_param = st.number_input("% Diện tích vá (P):", min_value=0.0, max_value=100.0, value=st.session_state.ai_p_param, step=0.1)
    st.metric(label="🚨 Tổng số ổ gà AI phát hiện", value=f"{st.session_state.ai_o_ga_count} Ổ gà")
    sv_param = st.number_input("Độ gồ ghề mặt đường (SV):", min_value=0.0, value=0.0, step=0.1)
    
    if loai_duong == "Đường nhựa (Mặt đường mềm)":
        rd_param = st.number_input("Chiều sâu lún vệt bánh xe trung bình (RD - mm):", min_value=0.0, value=0.0, step=0.1)
    else:
        rd_param = 0.0

    st.write("---")
    btn_tinh_psi = st.button("Tính điểm PSI", type="primary")

# ==========================================
# CỘT PHẢI: BẢN ĐỒ VỆ TINH & VẼ LỘ TRÌNH ĐƯỜNG ĐI
# ==========================================
with col2:
    st.header("🗺️ Bản đồ & Lộ trình thực tế")
    m = folium.Map(location=[current_lat, current_lon], zoom_start=16)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Vệ Tinh',
        name='Bản đồ vệ tinh',
        overlay=False
    ).add_to(m)
    folium.TileLayer('openstreetmap', name='Bản đồ đường phố').add_to(m)
    folium.LayerControl().add_to(m)
    
    if st.session_state.toa_do_bat_dau:
        folium.Marker(st.session_state.toa_do_bat_dau, popup="Điểm xuất phát", icon=folium.Icon(color='green', icon='play')).add_to(m)
    if st.session_state.toa_do_ket_thuc:
        folium.Marker(st.session_state.toa_do_ket_thuc, popup="Điểm kết thúc", icon=folium.Icon(color='red', icon='flag')).add_to(m)

    if len(st.session_state.lich_su_lo_trinh) > 1:
        folium.PolyLine(locations=st.session_state.lich_su_lo_trinh, color="cyan", weight=5, opacity=0.8).add_to(m)
        
    st_data = st_folium(m, width=700, height=450, key="map_tracker")

    st.write("---")
    st.subheader("🎯 Đánh giá & Chấm điểm hệ thống:")
    
    if btn_tinh_psi:
        if not ten_sv or not ma_doan:
            st.error("Vui lòng điền đầy đủ Tên SV và Mã đoạn đường trước khi chấm điểm!")
        elif not st.session_state.toa_do_ket_thuc:
            st.error("Bạn chưa kết thúc hành trình để lấy tọa độ tính toán lộ trình!")
        else:
            # Tính khoảng cách Haversine kiểm tra lộ trình
            lat1, lon1 = st.session_state.toa_do_ket_thuc
            lat2, lon2 = TOA_DO_MAU_LAT, TOA_DO_MAU_LON
            R = 6371000
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlam = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlam/2)**2
            khoang_cach = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
            dung_lo_trinh = khoang_cach <= 500.0
            
            # Tính điểm PSI theo AASHO
            log_sv = math.log10(1 + sv_param)
            sum_cp = c_param + p_param
            sqrt_cp = math.sqrt(sum_cp)
            
            if loai_duong == "Đường nhựa (Mặt đường mềm)":
                psi = 5.03 - (1.91 * log_sv) - (1.38 * (rd_param ** 2)) - (0.01 * sqrt_cp)
            else:
                psi = 5.41 - (1.78 * log_sv) - (0.09 * sqrt_cp)
            
            if st.session_state.ai_o_ga_count > 0:
                psi -= min(st.session_state.ai_o_ga_count * 0.05, 1.0)
            
            if psi > 5.0: psi = 5.0
            if psi < 0.0: psi = 0.0
            psi_rounded = round(psi, 1)
            
            st.metric(label="Chỉ số PSI Mặt đường", value=f"{psi_rounded} / 5.0")
            
            trang_thai_duong = "Rất tốt" if psi_rounded >= 4.0 else ("Trung bình" if psi_rounded >= 2.0 else "Hư hỏng nặng")
            if psi_rounded >= 4.0: st.success("🟢 Đường rất tốt, đi xe máy rất êm thuận.")
            elif psi_rounded >= 2.0: st.warning("🟡 Mặt đường có hư hỏng, rung lắc nhẹ.")
            else: st.error("🔴 Đường hư hỏng nặng, nguy hiểm do có nhiều ổ gà!")
            
            if dung_lo_trinh: st.success(f"✅ Đúng lộ trình quy định.")
            else: st.error(f"❌ Sai lộ trình quy định.")

            # ==========================================
            # 📊 BƯỚC 5: ĐÓNG GÓI VÀ XUẤT FILE EXCEL BÁO CÁO
            # ==========================================
            st.write("---")
            st.subheader("📊 Xuất file báo cáo nghiệm thu")
            
            # Tạo bảng dữ liệu bằng Pandas
            data_report = {
                "Thông số báo cáo": [
                    "Tên Sinh viên", "Mã đoạn đường", "Loại kết cấu mặt đường", 
                    "Thời gian kết thúc", "Tọa độ bắt đầu", "Tọa độ kết thúc", 
                    "Đúng lộ trình", "% Diện tích nứt (C)", "% Diện tích vá (P)", 
                    "Số lượng ổ gà AI", "Độ gồ ghề (SV)", "Chiều sâu lún vệt bánh (RD)", 
                    "Chỉ số PSI cuối cùng", "Trạng thái mặt đường"
                ],
                "Giá trị thực tế": [
                    ten_sv, ma_doan, loai_duong, 
                    st.session_state.thoi_gian_ket_thuc, str(st.session_state.toa_do_bat_dau), str(st.session_state.toa_do_ket_thuc),
                    "Đúng" if dung_lo_trinh else "Sai", c_param, p_param, 
                    st.session_state.ai_o_ga_count, sv_param, rd_param, 
                    psi_rounded, trang_thai_duong
                ]
            }
            
            df = pd.DataFrame(data_report)
            
            # Ghi dữ liệu vào bộ nhớ đệm dạng file Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Báo cáo PSI')
            
            # Tạo nút tải file Excel trực tiếp về máy tính của sinh viên
            st.download_button(
                label="📥 Tải xuống báo cáo kết quả (.xlsx)",
                data=buffer.getvalue(),
                file_name=f"Bao_cao_PSI_{ma_doan}_{ten_sv}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
