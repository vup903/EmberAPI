from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# 新增根目錄路由
@app.route('/')
def home():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从请求中提取 JSON 数据
        user_data = request.json

        # 模拟 AI 模型处理数据（这里我们只是返回接收到的数据作为示例）
        prediction_result = {
            "prediction": "这是一个模拟的预测结果",
            "confidence": 0.99
        }

        # 构建返回的 JSON 响应
        response = {
            "user_data": user_data,
            "prediction": prediction_result
        }

        # 返回 JSON 响应
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # 使用由环境变量 PORT 指定的端口，或默认为 5000
    app.run(host='0.0.0.0', port=port)
