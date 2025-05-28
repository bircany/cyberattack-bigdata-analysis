import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_categorical_distribution(df, column):
    """Kategorik değişkenlerin dağılımını görselleştirir."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y=column)
    plt.title(f'{column} Dağılımı')
    plt.tight_layout()
    plt.show()

def plot_numerical_distribution(df, column):
    """Sayısal değişkenlerin dağılımını görselleştirir."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f'{column} Dağılımı')
    plt.tight_layout()
    plt.show()

def plot_time_series(df, time_column, value_column):
    """Zaman serisi verilerini görselleştirir."""
    plt.figure(figsize=(12, 6))
    df.set_index(time_column)[value_column].plot()
    plt.title(f'{value_column} Zaman Serisi')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, numerical_columns):
    """Sayısal değişkenler arasındaki korelasyonu görselleştirir."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Korelasyon Matrisi')
    plt.tight_layout()
    plt.show()

# --- Veri Temizleme ve Ön İşleme Adımları ---

print("\n--- Veri Temizleme ve Ön İşleme Başlıyor ---")

# Veri setini yükle
file_path = r"C:\Users\birca\BüyükVeri Proje\data\cybersecurity_large_synthesized_data.csv"
df = pd.read_csv(file_path)

# timestamp sütununu datetime'a çevir
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Tekrar eden kayıtları kontrol et ve sil
print("\nTekrar Eden Kayıtlar Kontrol Ediliyor...")
initial_row_count = df.shape[0]
df.drop_duplicates(inplace=True) # inplace=True kullanarak DataFrame'i doğrudan güncelliyoruz
rows_removed = initial_row_count - df.shape[0]

if rows_removed > 0:
    print(f"- {rows_removed} adet tekrar eden kayıt bulundu ve silindi.")
else:
    print("- Tekrar eden kayıt bulunamadı.")

# 2. Eksik veri kontrolü ve doldurma
print("\nEksik Veriler Kontrol Ediliyor ve Dolduruluyor...")
missing_values_per_column = df.isnull().sum()
missing_columns = missing_values_per_column[missing_values_per_column > 0].index.tolist()

if len(missing_columns) > 0:
    total_missing_count = missing_values_per_column.sum()
    print(f"- {total_missing_count} adet eksik veri bulundu.")
    print("Eksik değerler sütun tiplerine göre dolduruluyor (sayısal: medyan, kategorik: mod)...")

    for column in missing_columns:
        if df[column].dtype in ['int64', 'float64']:
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
        else:
            mode_value = df[column].mode()[0] # Mod birden fazla değer döndürebilir, ilkini alıyoruz
            df[column].fillna(mode_value, inplace=True)

    print("Eksik veriler dolduruldu.")
else:
    print("- Eksik veri bulunamadı.")

# 3. Aykırı değer kontrolü (IQR yöntemi)
print("\nAykırı Değerler Kontrol Ediliyor ('attack_duration_min' sütunu için)...")
column_for_outlier_check = 'attack_duration_min'

if column_for_outlier_check in df.columns and df[column_for_outlier_check].dtype in ['int64', 'float64']:
    q1 = df[column_for_outlier_check].quantile(0.25)
    q3 = df[column_for_outlier_check].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = ((df[column_for_outlier_check] < lower_bound) | (df[column_for_outlier_check] > upper_bound)).sum()

    if outlier_count > 0:
        print(f"- {outlier_count} adet aykırı '{column_for_outlier_check}' değeri tespit edildi. İncelenebilir (Alt sınır: {lower_bound:.2f}, Üst sınır: {upper_bound:.2f}).")
    else:
        print(f"- '{column_for_outlier_check}' için aykırı değer bulunamadı.")
else:
    print(f"Uyarı: '{column_for_outlier_check}' sütunu bulunamadı veya sayısal değil, aykırı değer kontrolü atlandı.")

# 4. Temizlenmiş veri seti bilgileri
print("\nTemizlenmiş veri seti boyutu:", df.shape)
print("\nTemizlenmiş veri seti özeti:")
df.info()

# 5. Temizlenmiş veriyi kaydetme
output_directory = 'data'
output_filename = 'cleaned_cybersecurity_data.csv' 
output_path = f'{output_directory}\\{output_filename}'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"'{output_directory}' dizini oluşturuldu.")

df.to_csv(output_path, index=False)
print(f"\nTemizlenmiş veri '{output_path}' olarak kaydedildi.")

print("\n--- Veri Temizleme ve Ön İşleme Tamamlandı ---") 