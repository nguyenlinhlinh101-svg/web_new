from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Cho phép frontend HTML gọi API tự do

MODEL_PATH = 'student_model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("⚡ [SERVER] Khởi động thành công! Đã nạp bộ não AI 'student_model.pkl'.")
else:
    print("⚠️ [CẢNH BÁO] Không thấy file student_model.pkl. Vui lòng chạy lại file train.ipynb trước!")

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        
    if model is None:
        return jsonify({'error': 'Mô hình AI chưa được huấn luyện!'}), 400

    try:
        data = request.json
        # Sắp xếp các tham số đúng thứ tự đầu vào giống lúc huấn luyện
        input_data = np.array([[
            float(data['study_hours']),
            float(data['focus_score']),
            float(data['assignments']),
            float(data['attendance']),
            float(data['phone_hours']),
            float(data['stress_level']),
            float(data['sleep_hours'])
        ]])
        
        # Sử dụng AI để dự đoán điểm năng suất cho web OPTIME
        prediction = model.predict(input_data)[0]
        prediction = min(max(prediction, 15.0), 99.9) # Đảm bảo cận điểm thực tế
        
        return jsonify({'success': True, 'predicted_score': round(prediction, 2),'score':round(prediction,2)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def home():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


