from flask import Flask, render_template, request, jsonify
import math
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import requests
import json
import io
from datetime import datetime

app = Flask(__name__)

# Khởi tạo mô hình YOLO AI
print("Đang khởi tải mô hình YOLO AI...")
model = YOLO('yolov8n.pt')
print("Mô hình AI đã sẵn sàng!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        # Lấy dữ liệu khai báo kỹ thuật từ giao diện điện thoại gửi lên
        ten_sv = request.form.get('ten_sv')
        ma_doan = request.form.get('ma_doan')
        loai_duong = request.form.get('loai_duong')
        c_param = float(request.form.get('c_param', 0.0))
        p_param = float(request.form.get('p_param', 0.0))
        sv_param = float(request.form.get('sv_param', 0.0))
        rd_param = float(request.form.get('rd_param', 0.0))
        route_data = json.loads(request.form.get('route_data', '[]'))

        if len(route_data) < 2:
            return jsonify({"status": "error", "message": "Quãng đường di chuyển quá ngắn, chưa đủ dữ liệu GPS!"})

        # 1. AI Quét hình ảnh nhận diện và đếm số lượng ổ gà
        ai_o_ga_count = 0
        files = request.files.getlist('files[]')
        if files and files[0].filename != '':
            for file in files:
                if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_bytes = file.read()
                    img_np = np.array(Image.open(io.BytesIO(img_bytes)))
                    results = model(img_np, verbose=False)
                    for box in results[0].boxes:
                        if int(box.cls[0]) == 2:  # Đếm vật thể chỉ định (ví dụ xe/ổ gà tùy model)
                            ai_o_ga_count += 1

        # 2. Gọi API OSRM để bẻ cong tọa độ bám sát theo các tuyến phố thực tế
        lat_start, lon_start = route_data[0]
        lat_end, lon_end = route_data[-1]
        profile = "driving" if loai_duong == "nhua" else "foot"
        tong_quang_duong_m = 0.0
        osrm_route = route_data

        url = f"https://router.project-osrm.org/route/v1/{profile}/{lon_start},{lat_start};{lon_end},{lat_end}?overview=full&geometries=geojson"
        try:
            res = requests.get(url, timeout=5).json()
            if res.get("code") == "Ok":
                tong_quang_duong_m = float(res["routes"][0]["distance"])
                osrm_route = [[c[1], c[0]] for c in res["routes"][0]["geometry"]["coordinates"]]
        except:
            pass

        # 3. Thuật toán tính toán chỉ số chất lượng mặt đường PSI kỹ thuật
        gioi_han = 1000.0 if loai_duong == "nhua" else 500.0
        dung_lo_trinh = tong_quang_duong_m <= gioi_han
        
        log_sv = math.log10(1 + sv_param)
        sqrt_cp = math.sqrt(c_param + p_param)
        
        if loai_duong == "nhua":
            psi = 5.03 - 1.91 * log_sv - 1.38 * (rd_param ** 2) - 0.01 * sqrt_cp
        else:
            psi = 5.41 - 1.78 * log_sv - 0.09 * sqrt_cp
            
        if ai_o_ga_count > 0: 
            psi -= min(ai_o_ga_count * 0.05, 1.0)
            
        psi_rounded = round(max(0.0, min(5.0, psi)), 1)
        ket_luan = "Rất tốt và êm thuận" if psi_rounded >= 4.0 else ("Xuất hiện hư hỏng nhẹ" if psi_rounded >= 2.0 else "Hỏng hóc nghiêm trọng")

        # ----------------------------------------------------------------
        # 🚀 ĐƯỜNG DẪN GỬI KẾT QUẢ ĐÃ ĐƯỢC CHUYỂN ĐỔI THÀNH formResponse
        # ----------------------------------------------------------------
        form_url = "https://docs.google.com/forms/d/e/1FAIpQLSensmho-K5JH1x9oqY4_7V_UPqqQ1xrCsX6cRk_iqlqwFrhNw/formResponse"
        
        # Đổ chính xác dữ liệu tính toán từ AI và OSRM vào đúng các entry ID của bạn
        form_data = {
            'entry.304310822': ten_sv,
            'entry.1175553561': ma_doan,
            'entry.1009171500': "Đường Nhựa (Mặt đường mềm)" if loai_duong == "nhua" else "Đường BTXM (Mặt đường cứng)",
            'entry.1495392557': datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            'entry.1241991703': round(tong_quang_duong_m, 1),
            'entry.1100951947': "ĐÚNG TIÊU CHUẨN" if dung_lo_trinh else "VƯỢT QUÁ CHIỀU DÀI",
            'entry.983829169': ai_o_ga_count,
            'entry.2060331736': sv_param,
            'entry.994653011': psi_rounded,
            'entry.972794331': ket_luan
        }
        
        # Bắn dữ liệu ngầm lên Google Form bằng lệnh mạng POST
        requests.post(form_url, data=form_data)

        # Trả kết quả hiển thị thời gian thực về giao diện điện thoại sinh viên
        return jsonify({
            "status": "success", 
            "tong_quang_duong_m": round(tong_quang_duong_m, 1),
            "dung_lo_trinh": dung_lo_trinh, 
            "ai_o_ga_count": ai_o_ga_count,
            "psi_score": psi_rounded, 
            "osrm_route": osrm_route
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi xử lý hệ thống: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
