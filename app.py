from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json

app = Flask(__name__)

# 读取数据
csv_file_path = 'subset_10_percent2.csv'
if os.path.exists(csv_file_path):
    og_df = pd.read_csv(csv_file_path)
else:
    raise FileNotFoundError(f"{csv_file_path} 文件未找到，请确保文件存在并重试。")

# 定义血压分类函数
def categorize_bp(row):
    if row['bpavg_systolic_all_m_1_1a_v_1'] >= 140 or row['bpavg_diastolic_all_m_1_1a_v_1'] >= 90:
        return 'high'
    elif row['bpavg_systolic_all_m_1_1a_v_1'] < 90 or row['bpavg_diastolic_all_m_1_1a_v_1'] < 60:
        return 'low'
    else:
        return 'medium'

# 处理数据
if "bp_category" not in og_df.columns:
    og_df['bp_category'] = og_df.apply(categorize_bp, axis=1)
    og_df["age_new"] = og_df["age_new"].round(0)

# 定义模型训练函数
def train_model(df):
    X = df[["Rand_var1", "Rand_var2", "Rand_var3", "cigarettes_smoked_per_day", "number_of_cats", "bodyweight_kg_all_m_1_1a_v_1", "hours_exercise_per_week"]]
    y = df['bp_category']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model

# 计算概率变化的函数
def calculate_probability_change(original_proba, new_proba, target_class):
    return new_proba[target_class] - original_proba[target_class]

# 获取最佳特征的函数
def get_best_features(X_test, model):
    original_proba = model.predict_proba(X_test)[0]
    original_class = np.argmax(original_proba)

    if original_class != 1:  # 假设 '1' 为 'medium' 类，若当前分类不是 'medium'，则意味着可以改善
        healthier = True
    else:
        healthier = False

    largest_change = 0
    best_feature = None

    good_ranges = {
        "Rand_var1": (20, 40),
        "Rand_var2": (70, 85),
        "Rand_var3": (200, 295),
        "cigarettes_smoked_per_day": (0, 5),
        "number_of_cats": (1, 10),
        "bodyweight_kg_all_m_1_1a_v_1": (60, 80),
        "hours_exercise_per_week": (7, 16)
    }

    for feature in good_ranges:
        min_val, max_val = good_ranges[feature]
        if X_test[feature].iloc[0] < min_val or X_test[feature].iloc[0] > max_val:
            hypothetical_row = X_test.copy()
            hypothetical_row[feature] = (min_val + max_val) / 2
            new_proba = model.predict_proba(hypothetical_row)[0]
            change = calculate_probability_change(original_proba, new_proba, target_class=1)
            if change > largest_change:
                largest_change = change
                best_feature = feature

    return best_feature, healthier

# 根目录路由
@app.route('/')
def home():
    return "Hello, World!"

# 预测路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = request.json
        print(f"Received data: {user_data}")

        # 将JSON数据转换为DataFrame
        new_value = pd.DataFrame([user_data]).set_index(".id")
        print(f"DataFrame created from input: {new_value}")

        age_val = user_data["age"]
        gender_val = user_data["gender_1a_q_1"]
        height_val = user_data["bodylength_cm_all_m_1_1a_v_1"]

        # 加入年龄和身高的容差范围
        df = og_df[(og_df["age_new"].between(age_val - 5, age_val + 5)) &
                   (og_df["gender_1a_q_1"].str.upper() == gender_val.upper()) &  # 确保性别一致
                   (og_df['bodylength_cm_all_m_1_1a_v_1'].between(height_val - 5, height_val + 5))]

        if df.empty:
            print("No matching data found in the dataset.")
            return jsonify({"error": "No matching data found"}), 404

        model = train_model(df)
        print("Model trained successfully.")

        new_value2 = new_value[["Rand_var1", "Rand_var2", "Rand_var3", "cigarettes_smoked_per_day", "number_of_cats", "bodyweight_kg_all_m_1_1a_v_1", "hours_exercise_per_week"]]
        best_feature, healthier = get_best_features(new_value2, model)

        prediction = model.predict(new_value2)[0]
        print(f"Prediction: {prediction}, Best Feature: {best_feature}, Can Improve Class: {healthier}")

        response = {
            "user_data": user_data,
            "prediction": prediction,
            "best_feature": best_feature,
            "can_improve_class": healthier
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
