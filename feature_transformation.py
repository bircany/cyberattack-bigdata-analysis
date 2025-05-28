import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    from data_preprocessing import plot_categorical_distribution, plot_numerical_distribution, plot_time_series, plot_correlation_heatmap
    from feature_engineering import define_feature_lists # feature_engineering'den özellikleri import et
except ImportError as e:
    print(f"Hata: Gerekli fonksiyonlar import edilemedi. Hata: {e}")
    print("Lütfen data_preprocessing.py ve feature_engineering.py dosyalarının uygun şekilde konumlandırıldığından veya yüklendiğinden emin olun.")

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'

def perform_feature_transformation(df, numerical_features, categorical_features, target_column):
    """Sayısal ve kategorik özellikleri ölçeklendirir ve one-hot encode eder."""
    print("\n--- Özellik Dönüşümü Başlatılıyor ---")

    if df is None:
        print("Hata: DataFrame (df) mevcut değil. Dönüşüm yapılamadı.")
        return None, None, None

    # Özellikleri (X) ve hedef değişkeni (y) ayır
    # Hedef sütunu X'ten düşür
    columns_to_drop = ['outcome', 'timestamp', 'attacker_ip', 'target_ip']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    # Hedef sütun X'ten düşürülmeli
    if target_column in df.columns:
         existing_columns_to_drop.append(target_column)

    X = df.drop(columns=existing_columns_to_drop, axis=1, errors='ignore')

    if target_column in df.columns:
        y = df[target_column]
        # Hedef değişkeni sayısal hale getir (Label Encoding)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"Hedef sütun '{target_column}' Label Encoding ile sayısal hale getirildi.")
    else:
        print(f"Hata: Hedef sütun '{target_column}' DataFrame'de bulunamadı. Dönüşüm yapılamadı.")
        return None, None, None

    # Mevcut özellikleri kontrol et
    existing_numerical_features = [col for col in numerical_features if col in X.columns]
    existing_categorical_features = [col for col in categorical_features if col in X.columns]

    if not existing_numerical_features and not existing_categorical_features:
        print("Hata: Dönüşüm için geçerli sayısal veya kategorik özellik bulunamadı.")
        return None, None, None

    # Dönüşüm pipeline'ını oluştur
    transformers = []
    if existing_numerical_features:
        transformers.append(('num', StandardScaler(), existing_numerical_features))
    if existing_categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), existing_categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # Dönüşümü uygula
    try:
        X_transformed = preprocessor.fit_transform(X)
        print("Özellik dönüşümü tamamlandı!")

        # Özellik isimlerini al
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception as e:
            print(f"Uyarı: get_feature_names_out() başarısız oldu: {e}")
            feature_names = []
            feature_names.extend(existing_numerical_features)
            if 'cat' in preprocessor.named_transformers_:
                ohe = preprocessor.named_transformers_['cat']
                if hasattr(ohe, 'categories_'):
                    for i, col in enumerate(existing_categorical_features):
                        if i < len(ohe.categories_):
                            feature_names.extend([f"{col}_{cat}" for cat in ohe.categories_[i]])

        # Özellik isimlerinin sayısını kontrol et
        if len(feature_names) != X_transformed.shape[1]:
            print(f"Uyarı: Özellik adı sayısı ({len(feature_names)}) dönüştürülmüş sütun sayısı ({X_transformed.shape[1]}) ile eşleşmiyor.")
            feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]

        print(f"Toplam dönüştürülmüş özellik sayısı: {X_transformed.shape[1]}")

        # Dönüştürülmüş verinin ilk birkaç satırını göster
        X_transformed_df = pd.DataFrame(X_transformed.toarray(), columns=feature_names)
        print("\nDönüştürülmüş ilk 5 satır:")
        print(X_transformed_df.head())

        return X_transformed, y_encoded, feature_names

    except Exception as e:
        print(f"Hata oluştu: Özellik dönüşümü başarısız. Hata: {e}")
        return None, None, None

def analyze_feature_importance(X_transformed, y_encoded, feature_names):
    """Dönüştürülmüş özelliklerin hedef değişkenle korelasyonunu analiz eder."""
    print("\n--- Özellik Önem Analizi Başlatılıyor (Korelasyon) ---")

    if X_transformed is None or y_encoded is None or feature_names is None:
        print("Analiz için dönüştürülmüş özellikler veya hedef değişken mevcut değil.")
        return None

    # Korelasyonları hesapla
    X_df = pd.DataFrame(X_transformed.toarray(), columns=feature_names)
    correlations = X_df.corrwith(pd.Series(y_encoded, name='target'))

    correlation_with_target = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values
    })

    # Mutlak korelasyona göre sırala
    correlation_with_target['abs_correlation'] = correlation_with_target['correlation'].abs()
    sorted_correlations = correlation_with_target.sort_values(by='abs_correlation', ascending=False)

    # En önemli N özelliği göster
    top_n = 20
    print(f"\nEn önemli {top_n} özellik:")
    print(sorted_correlations.head(top_n).drop(columns='abs_correlation').reset_index(drop=True))

    return sorted_correlations

def save_transformed_data(X_transformed, y_encoded, feature_names, output_path='data/transformed_data.csv'):
    """Dönüştürülmüş özellikleri ve hedef değişkeni CSV dosyasına kaydeder."""
    print(f"\n--- Dönüştürülmüş Veri Seti Kaydediliyor ('{output_path}') ---")

    if X_transformed is None or y_encoded is None or feature_names is None:
        print("Kaydedilecek dönüştürülmüş özellikler veya hedef değişken mevcut değil.")
        return

    try:
        X_transformed_df = pd.DataFrame(X_transformed.toarray(), columns=feature_names)
        
        # Hedef değişkeni DataFrame'e ekle
        X_transformed_df['target'] = y_encoded

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        X_transformed_df.to_csv(output_path, index=False)
        print(f"Dönüştürülmüş veri seti başarıyla kaydedildi: '{output_path}'")
    except Exception as e:
        print(f"Hata oluştu: Dönüştürülmüş veri seti kaydedilemedi. Hata: {e}")

def save_correlation_results(correlation_results, output_path='correlation_results.csv'):
    """Korelasyon sonuçlarını CSV dosyasına kaydeder."""
    if correlation_results is None:
        print("Kaydedilecek korelasyon sonuçları mevcut değil.")
        return

    try:
        correlation_results.to_csv(output_path, index=False)
        print(f"Korelasyon sonuçları başarıyla kaydedildi: {output_path}")
    except Exception as e:
        print(f"Hata oluştu: Korelasyon sonuçları kaydedilemedi. Hata: {e}")

def define_feature_lists():
    """Kategorik ve sayısal özellik listelerini tanımlar."""
    categorical_features = [
        'attack_type', 'target_system', 'security_tools_used',
        'user_role', 'location', 'industry', 'mitigation_method',
        'attack_duration_category', 'damage_category', 'attack_severity'
    ]
    
    numerical_features = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'attacker_region', 'target_region',
        'attack_duration_min', 'data_compromised_GB',
        'response_time_min'
    ]
    
    return categorical_features, numerical_features

def create_preprocessing_pipeline(categorical_features, numerical_features):
    """Veri ön işleme pipeline'ı oluşturur."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def plot_feature_importance(importance_scores, feature_names):
    """Özellik önem skorlarını görselleştirir."""
    plt.figure(figsize=(12, 6))
    importance_df = pd.DataFrame({
        'Özellik': feature_names,
        'Önem Skoru': importance_scores
    }).sort_values('Önem Skoru', ascending=False)
    
    sns.barplot(data=importance_df, x='Önem Skoru', y='Özellik')
    plt.title('Özellik Önem Skorları', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, numerical_features):
    """Sayısal özellikler arasındaki korelasyonu görselleştirir."""
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Sayısal Özellikler Arasındaki Korelasyon', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Mühendislik yapılmış veriyi yükle
    engineered_data_path = r"C:\Users\birca\BüyükVeri Proje\data\engineered_cybersecurity_data.csv"
    
    if not os.path.exists(engineered_data_path):
        print(f"Hata: '{engineered_data_path}' dosyası bulunamadı!")
    else:
        df = pd.read_csv(engineered_data_path)
        
        # Özellik listelerini tanımla
        categorical_features, numerical_features = define_feature_lists()
        
        # Ön işleme pipeline'ı oluştur
        preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
        
        # Görselleştirmeler
        print("\n--- Özellik Dönüşümü Görselleştirmeleri ---")
        
        # Korelasyon matrisi
        plot_correlation_matrix(df, numerical_features)
        
        # Dönüştürülmüş veriyi kaydet
        output_path = r"C:\Users\birca\BüyükVeri Proje\data\transformed_cybersecurity_data.csv"
        df.to_csv(output_path, index=False)
        print(f"\nDönüştürülmüş veri kaydedildi: {output_path}") 