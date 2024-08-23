from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# 读取数据
og_df = pd.read_csv('bullshit2.csv')

# 定义分类函数
def categorize_bp(row):
    if row['bpavg_systolic'] >= 140 or row['bpavg_diastolic'] >= 90:
        return 'high'
    elif row['bpavg_systolic'] < 90 or row['bpavg_diastolic'] < 60:
        return 'low'
    else:
        return 'medium'

# 处理数据集
if "bp_category" not in og_df.columns:
    og_df['bp_category'] = og_df.apply(categorize_bp, axis=1)
    og_df["age"] = og_df["age"].round(0)

# 定义模型训练函数
def train_model(df):
    X = df[["income", "bodyweight", "education_years", "work_experience", "savings", "spending", "cigarettes_smoked_per_day", "number_of_cats", "savings_in_bank", "hours_exercise_per_week", "coffees_per_day"]]
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
def get_best_features(X_test, model, good_ranges):
    original_proba = model.predict_proba(X_test)[0]
    original_class = np.argmax(original_proba)
    
    if original_class != 1:
        healthier = True
    else:
        healthier = False
    
    largest_change = 0
    best_feature = None

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

# 定义根目录路由
@app.route('/')
def home():
    return "Hello, World!"

# 定义预测路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = request.json
        
        # 将JSON数据转换为DataFrame
        new_value = pd.DataFrame([user_data]).set_index(".id")
        age_val = user_data["age"]
        gender_val = user_data["gender"]
        height_val = user_data["bodylength"]

        df = og_df[(og_df["age"] == age_val) & 
                   (og_df["gender"] == gender_val) &
                   (og_df['bodylength'].between(height_val - 5, height_val + 5))]
        
        if df.empty:
            return jsonify({"error": "No matching data found"}), 404

        model = train_model(df)
        
        new_value2 = new_value[["income", "bodyweight", "education_years", "work_experience", "savings", "spending", "cigarettes_smoked_per_day", "number_of_cats", "savings_in_bank", "hours_exercise_per_week", "coffees_per_day"]]
        best_feature, healthier = get_best_features(new_value2, model, {
            "income":(10, 5006320),
            "bodyweight":(60, 80),
            "education_years":(10,20),
            "work_experience":(50, 60),
            "savings":(35126, 66468688),
            "spending":(3514, 666666),
            "cigarettes_smoked_per_day":(0, 5),
            "number_of_cats":(1, 10),
            "savings_in_bank":(10000, 10000000),
            "hours_exercise_per_week":(3.5, 14),
            "coffees_per_day":(2, 5),
        })

        prediction = model.predict(new_value2)[0]
        
        response = {
            "user_data": user_data,
            "prediction": prediction,
            "best_feature": best_feature,
            "can_improve_class": healthier
        }
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
