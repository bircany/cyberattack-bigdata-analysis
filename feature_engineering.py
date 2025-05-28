import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_time_features(df):
    """Zaman damgası sütunundan (timestamp) yeni zaman tabanlı özellikler türetir."""
    print("\n--- Zaman Tabanlı Özellikler Türetiliyor ---")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        print("Zaman tabanlı özellikler türetildi.")
        print("\nZaman tabanlı yeni özelliklere sahip ilk 5 satır:")
        print(df[['timestamp', 'hour', 'day_of_week', 'month', 'is_weekend']].head())
    else:
        print("Uyarı: 'timestamp' sütunu bulunamadı, zaman tabanlı özellikler türetilemedi.")

    print("--- Zaman Tabanlı Özellikler Türetme Tamamlandı ---")
    return df

def create_ip_region_features(df):
    """IP adreslerinin ilk oktetinden bölgesel (coğrafi) özellikler türetir."""
    print("\n--- IP Adreslerinden Bölgesel Özellikler Türetiliyor ---")

    def extract_ip_first_octet(ip_series):
        """IP adreslerinin ilk oktetini (sayı olarak) çıkarır."""
        return ip_series.astype(str).str.split('.').str[0].replace('', np.nan).astype(float).astype('Int64')

    attacker_ip_col = 'attacker_ip'
    target_ip_col = 'target_ip'

    if attacker_ip_col in df.columns and target_ip_col in df.columns:
        print(f"'{attacker_ip_col}' ve '{target_ip_col}' sütunlarından bölgesel özellikler türetiliyor...")
        df['attacker_region'] = extract_ip_first_octet(df[attacker_ip_col])
        df['target_region'] = extract_ip_first_octet(df[target_ip_col])

        print("Bölgesel özellikler türetildi!")
        print(f"Toplam {df['attacker_region'].nunique()} farklı saldırgan bölgesi ve {df['target_region'].nunique()} farklı hedef bölgesi tespit edildi.")

        print("\nSaldırgan ve hedef IP adresleri ile bölgesel kodları (İlk 5 örnek):")
        print(df[[attacker_ip_col, 'attacker_region', target_ip_col, 'target_region']].head())
    else:
        print(f"Uyarı: '{attacker_ip_col}' veya '{target_ip_col}' sütunlarından biri veya ikisi bulunamadı, bölgesel özellikler türetilemedi.")

    print("--- IP Adreslerinden Bölgesel Özellikler Türetme Tamamlandı ---")
    return df

def categorize_attack_duration(df):
    """Saldırı süresini (attack_duration_min) belirli aralıklara göre kategorize eder."""
    print("\n--- Saldırı Süresi Kategorileri Oluşturuluyor ---")

    duration_column = 'attack_duration_min'
    if duration_column in df.columns:
        bins = [0, 60, 300, 900, 3600, float('inf')]
        labels = ['çok_kısa', 'kısa', 'orta', 'uzun', 'çok_uzun']

        try:
            df['attack_duration_category'] = pd.cut(
                df[duration_column],
                bins=bins,
                labels=labels,
                right=True,
                include_lowest=True
            )
            print("Saldırı süresi kategorileri oluşturuldu!")
            print(f"'{duration_column}' sütununun ilk 5 örneği ve yeni kategori:")
            print(df[[duration_column, 'attack_duration_category']].head())
            print(f"'{duration_column}' kategori dağılımı:")
            print(df['attack_duration_category'].value_counts().sort_index())

        except Exception as e:
            print(f"Hata oluştu: Saldırı süresi kategorileri oluşturulamadı. Hata: {e}")
            df['attack_duration_category'] = None
    else:
        print(f"Uyarı: '{duration_column}' sütunu bulunamadı, saldırı süresi kategorileri oluşturulamadı.")
        df['attack_duration_category'] = None

    print("--- Saldırı Süresi Kategorileri Oluşturma Tamamlandı ---")
    return df

def categorize_damage(df):
    """Zarar miktarını (data_compromised_GB) niceleyicilere (quantiles) göre kategorize eder."""
    print("\n--- Zarar Miktarı Kategorileri Oluşturuluyor ---")

    damage_column = 'data_compromised_GB'
    if damage_column in df.columns and df[damage_column].dtype in ['int64', 'float64']:
        q = 5
        labels = ['çok_düşük', 'düşük', 'orta', 'yüksek', 'çok_yüksek']

        try:
            df['damage_category'] = pd.qcut(
                df[damage_column],
                q=q,
                labels=labels,
                duplicates='drop'
            )
            print("Zarar miktarı kategorileri oluşturuldu!")
            print(f"'{damage_column}' sütununun ilk 5 örneği ve yeni kategori:")
            print(df[[damage_column, 'damage_category']].head())
            print(f"'{damage_column}' kategori dağılımı:")
            print(df['damage_category'].value_counts().sort_index())

        except Exception as e:
            print(f"Hata oluştu: Zarar miktarı kategorileri oluşturulamadı. Hata: {e}")
            df['damage_category'] = None
    else:
        print(f"Uyarı: '{damage_column}' sütunu bulunamadı veya sayısal değil, zarar miktarı kategorileri oluşturulamadı.")
        df['damage_category'] = None

    print("--- Zarar Miktarı Kategorileri Oluşturma Tamamlandı ---")
    return df

def add_model_specific_features(df):
    """Model eğitimi için kullanılan ek özellikleri (oran, log, etkileşim, kare) ekler."""
    print("\n--- Modele Özel Ek Özellikler Ekleniyor ---")
    temp_df = df.copy() # Orijinal DataFrame'i etkilememek için kopya üzerinde çalışalım
    epsilon = 1e-6

    # Oran Özellikleri
    if 'data_compromised_GB' in temp_df.columns and 'attack_duration_min' in temp_df.columns:
        # Sayısal olmayanları NaN yapıp medyan ile doldur (input veride olabilir)
        temp_df['data_compromised_GB'] = pd.to_numeric(temp_df['data_compromised_GB'], errors='coerce').fillna(temp_df['data_compromised_GB'].median() if not temp_df['data_compromised_GB'].median() is np.nan else 0)
        temp_df['attack_duration_min'] = pd.to_numeric(temp_df['attack_duration_min'], errors='coerce').fillna(temp_df['attack_duration_min'].median() if not temp_df['attack_duration_min'].median() is np.nan else 0)
        
        temp_df['data_compromised_GB_attack_duration_ratio'] = temp_df['data_compromised_GB'] / (temp_df['attack_duration_min'] + epsilon)
        temp_df['data_compromised_GB_attack_duration_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        temp_df['data_compromised_GB_attack_duration_ratio'].fillna(temp_df['data_compromised_GB_attack_duration_ratio'].median() if not temp_df['data_compromised_GB_attack_duration_ratio'].median() is np.nan else 0, inplace=True)

    if 'attack_duration_min' in temp_df.columns and 'response_time_min' in temp_df.columns:
         # Sayısal olmayanları NaN yapıp medyan ile doldur (input veride olabilir)
        temp_df['attack_duration_min'] = pd.to_numeric(temp_df['attack_duration_min'], errors='coerce').fillna(temp_df['attack_duration_min'].median() if not temp_df['attack_duration_min'].median() is np.nan else 0)
        temp_df['response_time_min'] = pd.to_numeric(temp_df['response_time_min'], errors='coerce').fillna(temp_df['response_time_min'].median() if not temp_df['response_time_min'].median() is np.nan else 0)

        temp_df['attack_duration_min_response_time_ratio'] = temp_df['attack_duration_min'] / (temp_df['response_time_min'] + epsilon)
        temp_df['attack_duration_min_response_time_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        temp_df['attack_duration_min_response_time_ratio'].fillna(temp_df['attack_duration_min_response_time_ratio'].median() if not temp_df['attack_duration_min_response_time_ratio'].median() is np.nan else 0, inplace=True)

    # Logaritmik Dönüşüm Özellikleri
    log_features = [
        'data_compromised_GB', 'attack_duration_min', 'response_time_min'
    ]
    for col in log_features:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(temp_df[col].median() if not temp_df[col].median() is np.nan else 0) # Sayısal olmayanları ve NaN'ları doldur
            temp_df[f'{col}_log'] = np.log1p(temp_df[col])
            temp_df[f'{col}_log'].replace([np.inf, -np.inf], np.nan, inplace=True)
            temp_df[f'{col}_log'].fillna(temp_df[f'{col}_log'].median() if not temp_df[f'{col}_log'].median() is np.nan else 0, inplace=True)

    # Etkileşim ve Kare Terimleri
    if 'data_compromised_GB' in temp_df.columns and 'attack_duration_min' in temp_df.columns:
         temp_df['compromised_duration_interaction'] = temp_df['data_compromised_GB'] * temp_df['attack_duration_min']
    if 'attack_duration_min' in temp_df.columns and 'response_time_min' in temp_df.columns:
         temp_df['duration_response_interaction'] = temp_df['attack_duration_min'] * temp_df['response_time_min']
    if 'data_compromised_GB' in temp_df.columns and 'response_time_min' in temp_df.columns:
         temp_df['compromised_response_interaction'] = temp_df['data_compromised_GB'] * temp_df['response_time_min']

    if 'attack_duration_min' in temp_df.columns and 'response_time_min' in temp_df.columns:
        temp_df['duration_response_diff'] = temp_df['attack_duration_min'] - temp_df['response_time_min']
        temp_df['duration_response_abs_diff'] = np.abs(temp_df['attack_duration_min'] - temp_df['response_time_min'])

    for col in ['data_compromised_GB', 'attack_duration_min', 'response_time_min']:
        if col in temp_df.columns:
            temp_df[f'{col}_squared'] = temp_df[col]**2

    print("--- Modele Özel Ek Özellik Ekleme Tamamlandı ---")
    return temp_df

def plot_region_distribution(df):
    """En sık görülen 20 saldırgan bölgesinin dağılımını görselleştirir."""
    if 'attacker_region' in df.columns:
        plt.figure(figsize=(14, 7))
        top_n_regions = df['attacker_region'].value_counts().nlargest(20).index
        df_top_regions = df[df['attacker_region'].isin(top_n_regions)]
        
        sns.countplot(data=df_top_regions, y='attacker_region', order=top_n_regions, 
                     hue='attacker_region', legend=False, palette='viridis')
        
        plt.title('En Sık Görülen 20 Saldırgan Bölgesinin Dağılımı', fontsize=16)
        plt.xlabel('Saldırı Sayısı', fontsize=12)
        plt.ylabel('Saldırgan Bölgesi Kodu', fontsize=12)
        plt.tight_layout()
        plt.show()

def define_feature_lists():
    """Kategorik ve sayısal özellik listelerini tanımlar."""
    categorical_features = [
        'attack_type', 'target_system', 'security_tools_used',
        'user_role', 'location', 'industry', 'mitigation_method',
        'attack_duration_category', 'damage_category'
    ]
    
    numerical_features = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'attacker_region', 'target_region',
        'attack_duration_min', 'data_compromised_GB',
        'response_time_min',
        'attack_severity',
        'data_compromised_GB_attack_duration_ratio',
        'attack_duration_min_response_time_ratio',
        'data_compromised_GB_log',
        'attack_duration_min_log',
        'response_time_min_log',
        'data_compromised_GB_squared',
        'attack_duration_min_squared',
        'response_time_min_squared',
        'compromised_duration_interaction',
        'duration_response_interaction',
        'compromised_response_interaction',
        'duration_response_diff',
        'duration_response_abs_diff'
    ]
    
    return categorical_features, numerical_features

if __name__ == "__main__":
    # Temizlenmiş veriyi yükle
    cleaned_data_path = r"C:\Users\birca\BüyükVeri Proje\data\cleaned_cybersecurity_data.csv"
    
    if not os.path.exists(cleaned_data_path):
        print(f"Hata: '{cleaned_data_path}' dosyası bulunamadı!")
    else:
        df = pd.read_csv(cleaned_data_path)
        
        # Özellik mühendisliği adımlarını uygula
        df = create_time_features(df)
        df = create_ip_region_features(df)
        df = categorize_attack_duration(df)
        df = categorize_damage(df)
        
        # Bölgesel dağılımı görselleştir
        plot_region_distribution(df)
        
        # Özellik listelerini tanımla
        categorical_features, numerical_features = define_feature_lists()
        
        print("\nKategorik özellik listesi:", categorical_features)
        print("Sayısal özellik listesi:", numerical_features)
        
        # Mühendislik yapılmış veriyi kaydet
        output_path = r"C:\Users\birca\BüyükVeri Proje\data\engineered_cybersecurity_data.csv"
        df.to_csv(output_path, index=False)
        print(f"\nMühendislik yapılmış veri kaydedildi: {output_path}") 