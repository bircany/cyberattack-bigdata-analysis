# CyberAttack Big Data Analysis

Bu proje, siber saldırı verilerini büyük veri analizi tekniklerini kullanarak incelemeyi ve anlamayı amaçlamaktadır. Proje, veri ön işleme, özellik mühendisliği, veri dönüşümü ve makine öğrenmesi modeli eğitimi adımlarını içermektedir.

## Proje Amacı

*   Siber saldırı veri setini temizlemek ve analiz için hazırlamak.
*   Saldırı modellerini ve eğilimlerini ortaya çıkarmak için veri setini keşfetmek.
*   Makine öğrenmesi modelleri için anlamlı özellikler türetmek ve dönüştürmek.
*   Siber saldırıları tahmin etmek veya sınıflandırmak için modeller eğitmek (gelecekteki geliştirmeler dahil).

## Veri Seti

Bu projede, siber saldırı senaryolarını simüle eden sentetik bir büyük veri seti kullanılmıştır. Veri seti, saldırı türü, hedef sistem, zaman damgası, saldırgan/hedef IP, veri ihlali miktarı, saldırı süresi, kullanılan güvenlik araçları, kullanıcı rolü, konum, saldırı şiddeti, sektör, yanıt süresi ve azaltma yöntemi gibi bilgileri içermektedir.

Veri setinin boyutu: 100,000 kayıt ve 15 sütun.

## Proje Yapısı

Proje aşağıdaki ana Python dosyalarından oluşmaktadır:

*   `data_preprocessing.py`: Ham veri setinin yüklenmesi, temel analizlerin yapılması (istatistiksel özetler, dağılımlar, zaman serisi), eksik değerlerin doldurulması, duplike kayıtların silinmesi ve aykırı değerlerin yönetilmesi (capping) gibi veri temizleme ve ön işleme adımlarını içerir. Temizlenmiş veriyi bir CSV dosyasına kaydeder.
*   `feature_engineering.py`: Temizlenmiş veri setinden zaman bazlı özellikler (saat, gün, ay, hafta sonu), bölgesel özellikler (IP okteti) ve kategori bazlı özellikler (saldırı süresi, zarar miktarı kategorileri) gibi yeni ve anlamlı özellikler türetir. Mühendislik uygulanmış veriyi bir CSV dosyasına kaydeder.
*   `feature_transformation.py`: Mühendislik uygulanmış veri setindeki sayısal özellikleri ölçeklendirir (`StandardScaler`) ve kategorik özellikleri One-Hot Encoding (`OneHotEncoder`) ile sayısal formata d![Ekran görüntüsü 2025-05-28 032206](https://github.com/user-attachments/assets/8828a57b-615c-4088-98bf-e2303de1beaa)
önüştürür. Ayrıca, dönüştürülmüş özelliklerin hedef değişkenle korelasyonunu analiz eder ve görselleştirir. Dönüştürülmüş veri setini (özellikler ve hedef dahil) bir CSV dosyasına kaydeder.
*   `outcome_analysis.py`: (Bu dosyanın mevcut amacına göre bir açıklama eklenmeli. Örneğin: "Saldırı sonuçları (`outcome`) ile ilgili derinlemesine analizler ve görselleştirmeler yapar.")
*   `model_training.py`: (Bu dosyanın mevcut amacına göre bir açıklama eklenmeli. Örneğin: "Hazırlanmış veri seti üzerinde makine öğrenmesi modellerini eğitir, değerlendirir ve kaydeder.")

## Kurulum

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  Bu repoyu klonlayın:
    ```bash
    git clone https://github.com/KULLANICI_ADINIZ/cybersecurity-big-data-analysis.git
    ```
2.  Proje dizinine gidin:
    ```bash
    cd cybersecurity-big-data-analysis
    ```
3.  Gerekli Python kütüphanelerini yükleyin. Proje için bir `requirements.txt` dosyası oluşturmanız önerilir. İçine aşağıdaki kütüphaneleri ekleyebilirsiniz:
    ```
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    ```
    Ardından yüklemek için:
    ```bash
    pip install -r requirements.txt
    ```
4.  `data` klasörünün içine kullandığınız veri setini (`cybersecurity_large_synthesized_data.csv` gibi) yerleştirin veya kod içindeki dosya yolunu kendi veri setinize göre güncelleyin.

## Kullanım

Dosyaları sırayla çalıştırarak veri işleme ve analiz adımlarını gerçekleştirebilirsiniz:

1.  Veri ön işleme ve temizleme:
    ```bash
    python data_preprocessing.py
    ```
    Bu adım temizlenmiş veriyi `data/cleaned_cybersecurity_data.csv` olarak kaydedecektir.
2.  Özellik mühendisliği:
    ```bash
    python feature_engineering.py
    ```
    Bu adım mühendislik uygulanmış veriyi `data/engineered_cybersecurity_data.csv` olarak kaydedecektir.
3.  Özellik dönüşümü ve analiz:
    ```bash
    python feature_transformation.py
    ```
    Bu adım dönüştürülmüş veriyi `data/transformed_data.csv` ve korelasyon sonuçlarını `correlation_results.csv` olarak kaydedecektir. Ayrıca önemli özelliklerin korelasyon grafiklerini gösterecektir.
4.  Saldırı sonuçları analizi:
    ```bash
    python outcome_analysis.py
    ```
5.  Model eğitimi:
    ```bash
    python model_training.py
    ```

## Katkıda Bulunma

Projeye katkıda bulunmaktan mutluluk duyarız! Lütfen bir Pull Request göndermeden önce Issues kısmından önerilerinizi veya hataları bildirin.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Daha fazla bilgi için [LICENSE](LICENSE) dosyasına bakınız. (Eğer bir lisans dosyası ekleyecekseniz bu satırı ekleyin)
