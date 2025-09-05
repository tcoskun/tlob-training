# TLOB & LSTM Fiyat Tahmin Projesi

Bu proje, TLOB ve LSTM kullanarak fiyat tahminleme yapar. VectorBt ile porföy analizi yaparak LSTM ve TLOB çıktılarını karşılaştırır.

## 📁 Proje Çalışma Yapısı

```
tlob-training/
├── notebooks/
│   ├── analysis.ipynb                # TLOB Data Traning Notebook
│   ├── lstm_analysis.ipynb           # LSTM Data Traning Notebook
│   ├── portfolio_analysis.ipynb      # TLOB & LSTM VectorBt Backtesting
├── src/
│   ├── lstm_analyzer.py              # LSTM Data Training
│   ├── lstm_portfolio_analyzer.py    # LSTM VectorBt Backtesting
│   ├── data_loader.py                # TLOB Data Load
│   ├── data_preprocessor.py          # TLOB Data Pre Processor
│   ├── tlob_integration.py           # TLOB Library Integration
│   └── visualization.py              # TLOB Visualization
├── main.py                           # TLOB Data Traning & TLOB VectorBt Backtesting
├── lstm_main.py                      # LSTM Data Traning & LSTM VectorBt Backtesting
```
Not: models/tensorflow_results.pkl dosyası büyük olduğu için zip olarak eklendi

## 🚀 Kurulum

1. **Gereksinimleri yükleyin:**
```bash
pip install -r requirements.txt
```

2. **Veri dosyalarını `data/` klasörüne yerleştirin:**
   - CSV formatında LOB verileri
   - 10 seviyeli bid/ask fiyat ve hacim verileri

3. **Konfigürasyonu düzenleyin:**
   - `config/config.yaml` dosyasını ihtiyaçlarınıza göre ayarlayın

## 🏃‍♂️ Kullanım

### Hızlı Başlangıç

```bash
python main.py
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 
