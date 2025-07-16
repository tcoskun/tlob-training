import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os
import glob
import warnings
warnings.filterwarnings('ignore')

class LOBDataLoader:
    """
    Limit Order Book derinlik verilerini yükleyen ve ön işleyen generic sınıf
    """
    
    def __init__(self, file_path_or_dir: str):
        """
        Args:
            file_path_or_dir (str): CSV dosyasının yolu veya klasör
        """
        # Eğer klasörse, ilk CSV dosyasını bul
        if os.path.isdir(file_path_or_dir):
            csv_files = glob.glob(os.path.join(file_path_or_dir, '*.csv'))
            if not csv_files:
                raise FileNotFoundError(f"Klasörde CSV dosyası bulunamadı: {file_path_or_dir}")
            self.file_path = csv_files[0]
            print(f"Klasördeki ilk CSV dosyası kullanılacak: {self.file_path}")
        else:
            self.file_path = file_path_or_dir
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        CSV dosyasını yükler ve temel ön işleme yapar
        
        Returns:
            pd.DataFrame: İşlenmiş veri
        """
        print(f"Veri yükleniyor: {self.file_path}")
        
        # CSV dosyasını yükle
        self.data = pd.read_csv(self.file_path, sep=';', decimal=',')
        
        # DateTime sütununu datetime formatına çevir ve TR saatine çevir (GMT+3)
        self.data['DateTime'] = pd.to_datetime(self.data['DateTime'])
        # Eğer UTC ise TR saatine çevir (GMT+3)
        if self.data['DateTime'].dt.hour.iloc[0] < 10:  # UTC olduğunu varsay
            self.data['DateTime'] = self.data['DateTime'] + pd.Timedelta(hours=3)
        
        # Sütun isimlerini temizle
        self.data.columns = self.data.columns.str.strip()
        
        # Sayısal sütunları float'a çevir
        numeric_columns = [col for col in self.data.columns 
                          if any(x in col for x in ['Price', 'Volume', 'Ratio'])]
        
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Mid price hesapla ve ekle
        self.data['mid_price'] = (self.data['Level 1 Bid Price'] + self.data['Level 1 Ask Price']) / 2
        
        # Sembolü ayarla
        if 'Symbol' in self.data.columns:
            self.symbol = str(self.data['Symbol'].iloc[0])
        else:
            self.symbol = 'STOCK'
        
        print(f"Veri yüklendi. Satır sayısı: {len(self.data)}")
        print(f"Sütunlar: {list(self.data.columns)}")
        
        return self.data
    
    def get_lob_features(self) -> np.ndarray:
        """
        Limit Order Book özelliklerini numpy array olarak döndürür
        
        Returns:
            np.ndarray: LOB özellikleri [samples, features]
        """
        if self.data is None:
            self.load_data()
            
        # LOB özelliklerini topla
        lob_features = []
        
        for level in range(1, 6):
            bid_price = self.data[f'Level {level} Bid Price'].values
            bid_volume = self.data[f'Level {level} Bid Volume'].values
            ask_price = self.data[f'Level {level} Ask Price'].values
            ask_volume = self.data[f'Level {level} Ask Volume'].values
            
            lob_features.extend([bid_price, bid_volume, ask_price, ask_volume])
        
        # NaN değerleri 0 ile doldur
        lob_array = np.column_stack(lob_features)
        lob_array = np.nan_to_num(lob_array, nan=0.0)
        
        return lob_array
    
    def get_mid_price(self) -> np.ndarray:
        """
        Mid price hesaplar
        
        Returns:
            np.ndarray: Mid price değerleri
        """
        if self.data is None:
            self.load_data()
            
        bid_price = self.data['Level 1 Bid Price'].values
        ask_price = self.data['Level 1 Ask Price'].values
        
        mid_price = (bid_price + ask_price) / 2
        return np.nan_to_num(mid_price, nan=0.0)
    
    def get_price_changes(self, window: int = 10) -> np.ndarray:
        """
        Fiyat değişimlerini hesaplar
        
        Args:
            window (int): Pencere boyutu
            
        Returns:
            np.ndarray: Fiyat değişimleri
        """
        mid_price = self.get_mid_price()
        
        # Fiyat değişimi
        price_changes = np.diff(mid_price)
        
        # Pencere ortalaması
        if window > 1:
            price_changes = np.convolve(price_changes, np.ones(window)/window, mode='same')
        
        # İlk değer için 0 ekle
        price_changes = np.insert(price_changes, 0, 0)
        
        return price_changes
    
    def create_labels(self, horizon: int = 10, threshold: float = 0.0001) -> np.ndarray:
        """
        Fiyat trend etiketleri oluşturur
        
        Args:
            horizon (int): Tahmin ufku
            threshold (float): Trend eşiği
            
        Returns:
            np.ndarray: Etiketler (0: Düşüş, 1: Sabit, 2: Yükseliş)
        """
        mid_price = self.get_mid_price()
        
        labels = []
        
        for i in range(len(mid_price) - horizon):
            current_price = mid_price[i]
            future_price = mid_price[i + horizon]
            
            price_change = (future_price - current_price) / current_price
            
            if price_change > threshold:
                labels.append(2)  # Yükseliş
            elif price_change < -threshold:
                labels.append(0)  # Düşüş
            else:
                labels.append(1)  # Sabit
        
        # Son horizon kadar veri için etiket ekle
        labels.extend([1] * horizon)
        
        return np.array(labels)
    
    def get_summary_stats(self) -> dict:
        """
        Veri hakkında özet istatistikler döndürür
        
        Returns:
            dict: Özet istatistikler
        """
        if self.data is None:
            self.load_data()
            
        stats = {
            'total_rows': len(self.data),
            'date_range': (self.data['DateTime'].min(), self.data['DateTime'].max()),
            'symbols': self.data['Symbol'].unique().tolist(),
            'levels': self.data['Level'].unique().tolist(),
            'missing_values': self.data.isnull().sum().to_dict()
        }
        
        return stats 