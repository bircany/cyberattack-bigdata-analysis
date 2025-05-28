# Gerekli kütüphanelerin import edilmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import os 
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')  # Uyarıları kapat

# Tam dosya yolunu kullanma
file_path = r"C:\Users\birca\BüyükVeri Proje\data\engineered_cybersecurity_data_with_outcome_new.csv"

# Dosyanın var olup olmadığını kontrol etme
if not os.path.exists(file_path):
    print(f"Hata: '{file_path}' dosyası bulunamadı!")
    print("Mevcut çalışma dizini:", os.getcwd())
    print("Lütfen dosya yolunu kontrol edin.")
else:
    # Veri setini yükleme
    df = pd.read_csv(file_path)

    # outcome_new sınıf dağılımını kontrol etme
    print("\n--- outcome_new Sınıf Dağılımı ---")
    print(df['outcome_new'].value_counts())
    print("-----------------------------------")

    # Özellikleri (X) ve hedef değişkeni (y) ayırma
    # Model eğitiminde kullanmayacağımız sütunları düşürelim (eğitilirse mevcutsa)
    # attack_type sütununu da düşüreceğiz çünkü hedef değişken outcome
    columns_to_drop_for_model = ['attack_type', 'timestamp', 'attacker_ip', 'target_ip', 'outcome_new_numeric', 'outcome_new']
    existing_columns_to_drop_for_model = [col for col in columns_to_drop_for_model if col in df.columns]

    # Yeni hedef değişken outcome
    y = df['outcome']
    # outcome artık hedef değişken, X'ten düşürülmeli
    X = df.drop(columns=existing_columns_to_drop_for_model + ['outcome'], axis=1, errors='ignore')

    # outcome sınıf dağılımını kontrol etme
    print("\n--- outcome Sınıf Dağılımı (Orijinal Veri) ---")
    print(df['outcome'].value_counts())
    print("-----------------------------------------")

    # Hata giderme: Sayısal sütunlardaki sayısal olmayan değerleri temizleme
    # 'çok uzun' gibi değerleri NaN yapıp medyan ile dolduracağız.
    # NOT: Bu temizleme X DataFrame'i oluşturulduktan sonra yapılmalıdır.
    numerical_features_initial = [
        'data_compromised_GB', 'attack_duration_min', 'response_time_min'
    ]

    print("\n--- Sayısal Sütunları Temizleme ve Eksik Değerleri Doldurma ---")
    for col in numerical_features_initial:
        if col in X.columns:
            # Sayısal olmayanları NaN yap
            X[col] = pd.to_numeric(X[col], errors='coerce')
            # NaN değerleri medyan ile doldur
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"{col} sütunu temizlendi ve eksik değerler medyan ({median_val:.2f}) ile dolduruldu.")
        else:
            print(f"Uyarı: Sayısal özellik olarak belirtilen '{col}' sütunu veri setinde bulunamadı.")

    # Yeni Özellik Mühendisliği
    print("\n--- Yeni Özellikler Ekleniyor ---")
    # 1. Oran Özellikleri
    epsilon = 1e-6
    if 'data_compromised_GB' in X.columns and 'attack_duration_min' in X.columns:
        X['data_compromised_GB_attack_duration_ratio'] = X['data_compromised_GB'] / (X['attack_duration_min'] + epsilon)
        X['data_compromised_GB_attack_duration_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        X['data_compromised_GB_attack_duration_ratio'].fillna(X['data_compromised_GB_attack_duration_ratio'].median(), inplace=True)
        print("data_compromised_GB_attack_duration_ratio özelliği eklendi.")

    if 'attack_duration_min' in X.columns and 'response_time_min' in X.columns:
        X['attack_duration_min_response_time_ratio'] = X['attack_duration_min'] / (X['response_time_min'] + epsilon)
        X['attack_duration_min_response_time_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        X['attack_duration_min_response_time_ratio'].fillna(X['attack_duration_min_response_time_ratio'].median(), inplace=True)
        print("attack_duration_min_response_time_ratio özelliği eklendi.")

    # 2. Logaritmik Dönüşüm Özellikleri
    log_features = [
        'data_compromised_GB', 'attack_duration_min', 'response_time_min'
    ]
    for col in log_features:
        if col in X.columns:
            X[f'{col}_log'] = np.log1p(X[col])
            X[f'{col}_log'].replace([np.inf, -np.inf], np.nan, inplace=True)
            X[f'{col}_log'].fillna(X[f'{col}_log'].median(), inplace=True)
            print(f"{col}_log özelliği eklendi.")

    print("--- Yeni Özellik Ekleme Tamamlandı ---\n")

    # Model hiperparametrelerini güncelle
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    xgb = XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=200,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    lgbm = LGBMClassifier(
        num_leaves=31,
        learning_rate=0.1,
        n_estimators=200,
        objective='binary',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    lr = LogisticRegression(
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )

    # ÖN İŞLEME ADIMLARI
    print("\n" + "="*50)
    print(" VERİ ÖN İŞLEME ".center(50, '='))
    print("="*50 + "\n")

    # Kategorik ve sayısal özelliklerin tanımlanması
    # NOT: Yeni eklenen sayısal özellikler buraya dahil edilmeli
    categorical_features = [
        'target_system', 'security_tools_used',
        'user_role', 'location',
        'industry', 'mitigation_method'
    ]
    numerical_features = [
        'data_compromised_GB', 'attack_duration_min', 'response_time_min',
        'data_compromised_GB_attack_duration_ratio', # Yeni eklenen özellik
        'attack_duration_min_response_time_ratio', # Yeni eklenen özellik
        'data_compromised_GB_log', # Yeni eklenen özellik
        'attack_duration_min_log', # Yeni eklenen özellik
        'response_time_min_log', # Yeni eklenen özellik
        'data_compromised_GB_squared',
        'attack_duration_min_squared',
        'response_time_min_squared',
        'compromised_duration_interaction',
        'duration_response_interaction',
        'compromised_response_interaction',
        'duration_response_diff',
        'duration_response_abs_diff',
        'data_compromised_GB_normalized',
        'attack_duration_min_normalized',
        'response_time_min_normalized'
    ]

    # Ön işleme pipeline'ı
    # Sadece mevcut özellikleri transfomation'a dahil et
    existing_numerical_features = [col for col in numerical_features if col in X.columns]
    existing_categorical_features = [col for col in categorical_features if col in X.columns]

    transformers = []
    if existing_numerical_features:
         transformers.append(('num', StandardScaler(), existing_numerical_features))
    if existing_categorical_features:
         transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), existing_categorical_features))

    if not transformers:
        print("Hata: Model eğitimi için geçerli sayısal veya kategorik özellik bulunamadı.")
        # Buradan çıkış yapılabilir veya hata durumu yönetilebilir.
    else:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop') # remainder='drop' açıkça belirtilmeyen sütunları düşürür

        # Veriyi dönüştürme
        X_transformed = preprocessor.fit_transform(X)
        print("Özellik dönüşümü tamamlandı. Dönüşmüş veri boyutu:", X_transformed.shape)

        # Hedef değişkeni kodlama
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(f"Hedef değişken kodlandı. Sınıflar: {list(le.classes_)}")


        # Dönüştürülmüş özellik isimlerini alma
        # sklearn 0.23 ve sonrası için:
        try:
             feature_names = preprocessor.get_feature_names_out()
        except Exception as e:
             print(f"Uyarı: preprocessor.get_feature_names_out() başarısız oldu: {e}")
             print("Özellik isimleri elle oluşturuluyor, bu hatalı olabilir.")
             # Elle oluşturma (yedek yöntem, doğru çalışmayabilir)
             feature_names = existing_numerical_features.copy()
             if 'cat' in preprocessor.named_transformers_:
                 ohe = preprocessor.named_transformers_['cat']
                 if hasattr(ohe, 'categories_'):
                      for i, cat_col in enumerate(existing_categorical_features):
                           if i < len(ohe.categories_):
                                feature_names.extend([f"{cat_col}_{cat}" for cat in ohe.categories_[i]])
                 else:
                      print("Uyarı: OneHotEncoder kategorileri bulunamadı.")


        print(f"Dönüştürülmüş özellik isimleri sayısı: {len(feature_names)}")
        if len(feature_names) != X_transformed.shape[1]:
             print(f"Hata: Özellik adı sayısı ({len(feature_names)}) ile dönüştürülmüş sütun sayısı ({X_transformed.shape[1]}) eşleşmiyor. Özellik önemi grafikleri hatalı olabilir.")
             # Eşleşmezse generic isimler kullanalım ki kod çalışsın
             feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]


        # Veriyi eğitim ve test setlerine ayırma
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Eğitim veri boyutu: {X_train.shape}")
        print(f"Test veri boyutu: {X_test.shape}")

        # Orijinal eğitim verisini kullan
        X_train_used, y_train_used = X_train, y_train

        # RANDOM FOREST MODELİ
        print("\n" + "="*50)
        print(" RANDOM FOREST MODELİ ".center(50, '='))
        print("="*50 + "\n")

        # Model hiperparametrelerini güncelle
        rf.fit(X_train_used, y_train_used)

        # Tahminler ve değerlendirme
        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest Test Doğruluğu: {acc_rf:.4f}")

        print("\nKarışıklık Matrisi:")
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        print(pd.DataFrame(cm_rf, index=le.classes_, columns=le.classes_))

        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

        # LOJİSTİK REGRESYON MODELİ
        print("\n" + "="*50)
        print(" LOJİSTİK REGRESYON MODELİ ".center(50, '='))
        print("="*50 + "\n")

        # Lojistik Regresyon için
        lr.fit(X_train_used, y_train_used)

        # Tahminler ve değerlendirme
        y_pred_lr = lr.predict(X_test)
        acc_lr = accuracy_score(y_test, y_pred_lr)
        print(f"Lojistik Regresyon Test Doğruluğu: {acc_lr:.4f}")

        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

        # XGBOOST MODELİ
        print("\n" + "="*50)
        print(" XGBOOST MODELİ ".center(50, '='))
        print("="*50 + "\n")

        # Model hiperparametrelerini güncelle
        xgb.fit(X_train_used, y_train_used)

        y_pred_xgb = xgb.predict(X_test)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        print(f"XGBoost Test Doğruluğu: {acc_xgb:.4f}")

        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

        # LightGBM MODELİ
        print("\n" + "="*50)
        print(" LIGHTGBM MODELİ ".center(50, '='))
        print("="*50 + "\n")

        # Model hiperparametrelerini güncelle
        lgbm.fit(X_train_used, y_train_used)

        y_pred_lgbm = lgbm.predict(X_test)
        acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
        print(f"LightGBM Test Doğruluğu: {acc_lgbm:.4f}")

        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred_lgbm, target_names=le.classes_))

        print("\nModel eğitimi ve değerlendirmesi tamamlandı.")

        # Eğitilmiş modelleri ve LabelEncoder'ı kaydetme
        try:
            model_dir = "models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # Random Forest modelini kaydet
            rf_model_path = os.path.join(model_dir, "random_forest_model_outcome.joblib")
            joblib.dump(rf, rf_model_path)
            print(f"Random Forest modeli kaydedildi: {rf_model_path}")

            # Lojistik Regresyon modelini kaydet
            lr_model_path = os.path.join(model_dir, "logistic_regression_model_outcome.joblib")
            joblib.dump(lr, lr_model_path)
            print(f"Lojistik Regresyon modeli kaydedildi: {lr_model_path}")

            # XGBoost modelini kaydet
            xgb_model_path = os.path.join(model_dir, "xgboost_model_outcome.joblib")
            joblib.dump(xgb, xgb_model_path)
            print(f"XGBoost modeli kaydedildi: {xgb_model_path}")

            # LightGBM modelini kaydet
            lgbm_model_path = os.path.join(model_dir, "lightgbm_model_outcome.joblib")
            joblib.dump(lgbm, lgbm_model_path)
            print(f"LightGBM modeli kaydedildi: {lgbm_model_path}")

            # LabelEncoder'ı kaydet
            le_path = os.path.join(model_dir, "outcome_label_encoder.joblib")
            joblib.dump(le, le_path)
            print(f"LabelEncoder kaydedildi: {le_path}")

            # Preprocessor'ı kaydet
            preprocessor_path = os.path.join(model_dir, "preprocessor_for_outcome.joblib")
            joblib.dump(preprocessor, preprocessor_path)
            print(f"Preprocessor kaydedildi: {preprocessor_path}")

        except Exception as e:
            print(f"Model veya LabelEncoder kaydedilirken bir hata oluştu: {e}") 