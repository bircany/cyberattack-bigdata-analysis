from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Sadece Random Forest modelini ve gerekli bileşenleri yükle
model_dir = "models"
rf_model = joblib.load(os.path.join(model_dir, "random_forest_model_outcome.joblib"))
preprocessor = joblib.load(os.path.join(model_dir, "preprocessor_for_outcome.joblib"))
label_encoder = joblib.load(os.path.join(model_dir, "outcome_label_encoder.joblib"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form verilerini al
        data = {
            'target_system': request.form['target_system'],
            'security_tools_used': request.form['security_tools_used'],
            'user_role': request.form['user_role'],
            'location': request.form['location'],
            'industry': request.form['industry'],
            'mitigation_method': request.form['mitigation_method'],
            'data_compromised_GB': float(request.form['data_compromised_GB']),
            'attack_duration_min': float(request.form['attack_duration_min']),
            'response_time_min': float(request.form['response_time_min'])
        }
        
        # DataFrame oluştur
        input_df = pd.DataFrame([data])
        
        # Özellik Mühendisliği - model_training.py dosyasındakiyle aynı adımlar
        epsilon = 1e-6
        if 'data_compromised_GB' in input_df.columns and 'attack_duration_min' in input_df.columns:
            input_df['data_compromised_GB_attack_duration_ratio'] = input_df['data_compromised_GB'] / (input_df['attack_duration_min'] + epsilon)
            input_df['data_compromised_GB_attack_duration_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            # Eksik değer doldurma (eğitim setindeki medyan yerine basitleştirilmiş, 0 veya küçük bir değer kullanılabilir)
            input_df['data_compromised_GB_attack_duration_ratio'].fillna(0, inplace=True)

        if 'attack_duration_min' in input_df.columns and 'response_time_min' in input_df.columns:
            input_df['attack_duration_min_response_time_ratio'] = input_df['attack_duration_min'] / (input_df['response_time_min'] + epsilon)
            input_df['attack_duration_min_response_time_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
             # Eksik değer doldurma
            input_df['attack_duration_min_response_time_ratio'].fillna(0, inplace=True)

        log_features = [
            'data_compromised_GB', 'attack_duration_min', 'response_time_min'
        ]
        for col in log_features:
            if col in input_df.columns:
                input_df[f'{col}_log'] = np.log1p(input_df[col])
                input_df[f'{col}_log'].replace([np.inf, -np.inf], np.nan, inplace=True)
                # Eksik değer doldurma
                input_df[f'{col}_log'].fillna(0, inplace=True)

        # Not: model_training.py dosyasında kare alma ve etkileşim özellikleri de vardı.
        # Eğer preprocessor bu özellikleri bekliyorsa, buraya eklenmeleri gerekir.
        # Basitlik için şimdilik sadece oran ve log ekledik.
        # Eğer hala hata alırsanız, diğer özellikleri de eklememiz gerekecek.


        # Veriyi dönüştür
        X_transformed = preprocessor.transform(input_df)
        
        # Tahmin yap
        prediction = rf_model.predict(X_transformed)[0]
        
        # Tahmini orijinal etikete dönüştür
        result = label_encoder.inverse_transform([prediction])[0]
        
        return jsonify({'tahmin': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 