# TLOB Fiyat Tahmin Projesi

Bu proje, Limit Order Book (LOB) verilerini kullanarak **5 günlük fiyat tahmini** yapan sadeleştirilmiş bir TLOB (Time-weighted Limit Order Book) modelidir.

## 🎯 Proje Amacı

- 5 seviyeli LOB verilerini analiz eder
- **5 günlük fiyat yönü tahmini** yapar (Düşüş/Sabit/Yükseliş)
- Gün gün görsel tahmin raporu sunar
- Güven skorları ile tahmin kalitesini gösterir
- Sadeleştirilmiş görselleştirmeler sunar
- Hızlı ve etkili model eğitimi sağlar

## 📁 Proje Yapısı

```
tlob-training/
├── config/
│   └── config.yaml          # Konfigürasyon dosyası
├── data/
│   └── *.csv               # LOB veri dosyaları
├── models/                 # Eğitilmiş modeller
├── src/
│   ├── data_loader.py      # Veri yükleme
│   ├── data_preprocessor.py # Veri ön işleme
│   ├── model_trainer.py    # Model eğitimi + 5 günlük tahmin
│   ├── simple_tlob.py      # TLOB modeli
│   └── visualization.py    # Görselleştirme + 5 günlük grafik
├── main.py                 # Ana çalıştırma dosyası
└── requirements.txt        # Gereksinimler
```

## 🚀 Kurulum

1. **Gereksinimleri yükleyin:**
```bash
pip install -r requirements.txt
```

2. **Veri dosyalarını `data/` klasörüne yerleştirin:**
   - CSV formatında LOB verileri
   - 5 seviyeli bid/ask fiyat ve hacim verileri

3. **Konfigürasyonu düzenleyin:**
   - `config/config.yaml` dosyasını ihtiyaçlarınıza göre ayarlayın

## 🏃‍♂️ Kullanım

### Hızlı Başlangıç

```bash
python main.py
```

### Adım Adım İşlem

1. **Veri Yükleme:** LOB verilerini yükler ve temel istatistikleri gösterir
2. **Veri Ön İşleme:** TLOB özelliklerini hesaplar ve sequence'ler oluşturur
3. **Model Eğitimi:** SimpleTLOB modelini eğitir
4. **Değerlendirme:** Test seti üzerinde performansı ölçer
5. **Temel Görselleştirmeler:** Fiyat evrimi, eğitim geçmişi, tahmin sonuçları
6. **5 Günlük Tahmin:** Gün gün fiyat yönü tahmini ve görselleştirme
7. **Ertesi Gün Tahmini:** Tek günlük detaylı tahmin

## 📊 Çıktılar

### Temel Görselleştirmeler
- **Fiyat Evrimi:** Zaman içinde fiyat değişimi
- **Eğitim Geçmişi:** Loss ve accuracy grafikleri
- **Tahmin Sonuçları:** Confusion matrix ve sınıf dağılımları

### 🆕 5 Günlük Tahmin Görselleştirmesi
- **Fiyat Grafiği:** 5 günlük simüle edilmiş fiyat hareketi
- **Günlük Tahminler:** Her gün için yön tahmini (Düşüş/Sabit/Yükseliş)
- **Güven Skorları:** Her tahmin için güvenilirlik oranı
- **Detaylı Tablo:** Gün gün fiyat ve değişim yüzdesi

### Model Performansı
- Accuracy, Precision, Recall, F1-Score
- Sınıf bazında performans analizi
- 5 günlük tahmin sonuçları

## ⚙️ Konfigürasyon

`config/config.yaml` dosyasında ayarlayabileceğiniz parametreler:

```yaml
model:
  sequence_length: 10      # Kaç günlük veri kullanılacak
  prediction_horizon: 1    # Ertesi gün tahmini
  forecast_days: 5         # 5 günlük tahmin
  test_size: 0.2          # Test seti oranı

training:
  epochs: 250              # Eğitim epoch sayısı
  batch_size: 64          # Batch boyutu
  learning_rate: 0.001    # Öğrenme oranı

visualization:
  show_confidence: true   # Güven skorlarını göster
```

## 📈 Model Özellikleri

### SimpleTLOB Modeli
- **Giriş:** 5 seviyeli LOB verileri (fiyat + hacim)
- **Özellikler:** TLOB hesaplamaları (spread, imbalance, vb.)
- **Çıkış:** 3 sınıf (Düşüş/Sabit/Yükseliş)
- **Mimari:** Transformer tabanlı

### 🆕 5 Günlük Tahmin Sistemi
- **Sequential Prediction:** Her gün için ayrı tahmin
- **Sequence Update:** Tahmin sonucuna göre veri güncelleme
- **Confidence Scoring:** Her tahmin için güven skoru
- **Price Simulation:** Tahminlere göre fiyat simülasyonu

### TLOB Özellikleri
- **Spread:** Bid-ask farkı
- **Imbalance:** Bid-ask hacim dengesizliği
- **Depth:** Order book derinliği
- **Pressure:** Fiyat baskısı göstergeleri

## 🔧 Gereksinimler

- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 📝 Notlar

- Proje sadeleştirilmiş versiyondur
- **5 günlük tahmin** odaklıdır
- Gün gün görsel raporlama sunar
- Güven skorları ile tahmin kalitesini gösterir
- Gereksiz görseller kaldırılmıştır
- Hızlı eğitim için optimize edilmiştir

## 🎯 Örnek Çıktı

```
============================================================
AKBNK 5 GÜNLÜK TAHMİN DETAYI
============================================================
Başlangıç Fiyatı: 45.2500
============================================================
1. Gün   | Yükseliş | 46.1550 |  +2.0% (Güven: 78.5%)
2. Gün   | Sabit    | 46.1550 |  +0.0% (Güven: 65.2%)
3. Gün   | Düşüş    | 45.2319 |  -2.0% (Güven: 72.1%)
4. Gün   | Yükseliş | 46.1365 |  +2.0% (Güven: 69.8%)
5. Gün   | Sabit    | 46.1365 |  +0.0% (Güven: 71.3%)
============================================================
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 