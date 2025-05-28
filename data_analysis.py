import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')  # Tüm uyarıları kapat

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_categorical_distribution(df, column, title, palette):
    """Kategorik değişkenlerin dağılımını çubuk grafik ile görselleştirir."""
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x=column, hue=column, palette=palette, legend=False)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_numerical_distribution(df, column, title, color):
    """Sayısal değişkenlerin dağılımını histogram ile görselleştirir."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, bins=30, color=color)
    plt.title(title, fontsize=14)
    plt.xlabel(column)
    plt.tight_layout()
    plt.show()

def plot_time_series(df, time_column, value_column, title, color):
     """Zaman serisi verilerini çizgi grafik ile görselleştirir."""
     df[time_column] = pd.to_datetime(df[time_column])
     daily_data = df.groupby(df[time_column].dt.date).size()
     
     plt.figure(figsize=(15, 6))
     daily_data.plot(color=color)
     plt.title(title, fontsize=14)
     plt.xlabel('Tarih')
     plt.ylabel('Sayı') # Bu kısım genel tutuldu, duruma göre değişebilir
     plt.grid(True, alpha=0.3)
     plt.tight_layout()
     plt.show()

def plot_correlation_heatmap(df, numerical_cols, title, cmap):
    """Sayısal değişkenler arasındaki korelasyonu ısı haritası ile görselleştirir."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap=cmap, center=0)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def load_and_check_data(file_path):
    """Veri setini yükler ve temel kontrolleri yapar."""
    if not os.path.exists(file_path):
        print(f"Hata: '{file_path}' dosyası bulunamadı!")
        print("Mevcut çalışma dizini:", os.getcwd())
        print("Lütfen dosya yolunu kontrol edin.")
        return None
    
    df = pd.read_csv(file_path)
    
    # timestamp sütununu datetime'a çevir
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\nVeri seti boyutu: {df.shape}")
    print("\nVeri seti sütunları:", df.columns.tolist())
    print("\nVeri seti özeti:")
    df.info()
    return df

def analyze_data(df):
    """Veri setinin temel analizini yapar."""
    print("\n" + "="*50)
    print(" VERİ ANALİZİ ".center(50, '='))
    print("="*50 + "\n")
    
    print("\nSayısal değişkenlerin istatistiksel özeti:")
    print(df.describe())
    
    # Saldırı türlerinin dağılımı
    plot_categorical_distribution(df, 'attack_type', 'Saldırı Türlerinin Dağılımı', 'viridis')

    # Saldırı şiddetinin dağılımı
    plot_categorical_distribution(df, 'attack_severity', 'Saldırı Şiddetinin Dağılımı', 'plasma')

    # Sektörlere göre saldırı dağılımı
    plot_categorical_distribution(df, 'industry', 'Sektörlere Göre Saldırı Dağılımı', 'magma')

    # Saldırı süresinin dağılımı
    plot_numerical_distribution(df, 'attack_duration_min', 'Saldırı Süresinin Dağılımı', 'skyblue')

    # Zaman serisi analizi
    plot_time_series(df, 'timestamp', None, 'Günlük Saldırı Sayısı', 'purple')

    # Sayısal değişkenler arasındaki korelasyon
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    plot_correlation_heatmap(df, numerical_cols, 'Sayısal Değişkenler Arasındaki Korelasyon', 'coolwarm')

def clean_data(df):
    """Veri setini temizler."""
    print("\n" + "="*50)
    print(" VERİ TEMİZLEME VE ÖN İŞLEME ".center(50, '='))
    print("="*50 + "\n")
    
    print("Tekrar Eden Kayıtlar Kontrol Ediliyor...")
    # Duplike kayıtları kontrol et ve temizle
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"- {duplicate_count} adet tekrar eden kayıt bulundu ve temizlendi.")
        df = df.drop_duplicates()
    else:
        print("- Tekrar eden kayıt bulunamadı.")

    print("\nEksik Veriler Kontrol Ediliyor ve Dolduruluyor...")
    # Eksik değerleri kontrol et ve doldur
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("\nEksik değerler:")
        print(missing_values[missing_values > 0])
        
        # Sayısal sütunlar için medyan, kategorik sütunlar için mod kullan
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        print("- Eksik değerler dolduruldu.")
    else:
        print("- Eksik veri bulunamadı.")

    print("\nAykırı Değerler Kontrol Ediliyor ve Yönetiliyor...")
    # Aykırı değerleri kontrol et, raporla ve yönet (capping)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outliers_managed = False

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        print(f"\n{col} sütunu için aykırı değer analizi:")
        if len(outliers) > 0:
            outliers_managed = True
            print(f"- Aykırı değer sayısı: {len(outliers)} ({len(outliers) / len(df) * 100:.2f}%)")
            print(f"- Minimum aykırı değer: {outliers.min():.2f}")
            print(f"- Maksimum aykırı değer: {outliers.max():.2f}")
            print(f"- Alt sınır (IQR Metodu): {lower_bound:.2f}")
            print(f"- Üst sınır (IQR Metodu): {upper_bound:.2f}")
            
            # Aykırı değerleri uç sınırlara sabitleme (Capping)
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            print(f"- Aykırı değerler [{lower_bound:.2f}, {upper_bound:.2f}] aralığına sabitlendi (Capping).")

        else:
            print(f"- Aykırı değer bulunamadı.")

    if not outliers_managed:
        print("- Tüm sayısal sütunlarda aykırı değer bulunamadı.")

    # Temizlenmiş veri setinin özeti
    print("\nTemizlenmiş veri seti boyutu:", df.shape)
    print("\nTemizlenmiş veri seti özeti:")
    df.info()

    return df

def save_cleaned_data(df, output_path):
    """Temizlenmiş veriyi kaydeder."""
    # Çıktı dizininin var olduğundan emin ol
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_path, index=False)
    print(f"\nTemizlenmiş veri seti kaydedildi: {output_path}")

if __name__ == "__main__":
    # Tam dosya yolunu kullanma
    file_path = r"C:\Users\birca\BüyükVeri Proje\data\cybersecurity_large_synthesized_data.csv"
    
    # Veriyi yükle
    df = load_and_check_data(file_path)
    
    if df is not None:
        # Veriyi analiz et
        analyze_data(df)
        
        # Veriyi temizle
        df_cleaned = clean_data(df)
        
        # Temizlenmiş veriyi kaydet
        output_path = r"C:\Users\birca\BüyükVeri Proje\data\cleaned_cybersecurity_data.csv"
        save_cleaned_data(df_cleaned, output_path) 