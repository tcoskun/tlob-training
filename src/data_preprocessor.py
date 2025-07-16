import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import sys
import os

class LOBDataPreprocessor:
    """
    Generic LOB verisi için TLOB kütüphanesi uyumlu ön işleme sınıfı
    """
    
    def __init__(self, sequence_length: int = 50, prediction_horizon: int = 10):
        """
        Args:
            sequence_length (int): Giriş sekans uzunluğu
            prediction_horizon (int): Tahmin ufku
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        
    def prepare_tlob_data(self, data_loader, test_size: float = 0.2, validation_size: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        TLOB modeli için veriyi hazırlar
        
        Args:
            data_loader: LOBDataLoader instance
            test_size: Test seti oranı (varsayılan: 0.2)
            validation_size: Validation seti oranı (varsayılan: 0.1)
            random_state: Tekrarlanabilirlik için seed
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (X_train, X_test, y_train, y_test)
        """
        print("TLOB için veri hazırlanıyor...")
        
        # LOB özelliklerini al
        lob_features = data_loader.get_lob_features()
        
        # Etiketleri oluştur
        labels = data_loader.create_labels(horizon=self.prediction_horizon)
        
        # Veriyi normalize et
        lob_features_scaled = self.scaler.fit_transform(lob_features)
        
        # Sekans verisi oluştur
        X, y = self.create_sequences(lob_features_scaled, labels)
        
        # Train/test split - konfigürasyondan alınan değerlerle
        total_samples = len(X)
        test_samples = int(test_size * total_samples)
        train_samples = total_samples - test_samples
        
        # Rastgele karıştırma için seed ayarla
        np.random.seed(random_state)
        indices = np.random.permutation(total_samples)
        
        # Train ve test setlerini ayır
        train_indices = indices[:train_samples]
        test_indices = indices[train_samples:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        print(f"Veri hazırlandı:")
        print(f"  Toplam örnek: {total_samples}")
        print(f"  Train örnek: {train_samples} (%{100-test_size*100:.0f})")
        print(f"  Test örnek: {test_samples} (%{test_size*100:.0f})")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sekans verisi oluşturur
        
        Args:
            features (np.ndarray): Özellikler
            labels (np.ndarray): Etiketler
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y)
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def prepare_lobster_format(self, data_loader) -> None:
        """
        LOBSTER formatında veri hazırlar (TLOB kütüphanesi için)
        
        Args:
            data_loader: LOBDataLoader instance
        """
        print("LOBSTER formatında veri hazırlanıyor...")
        
        # Veriyi yükle
        df = data_loader.load_data()
        
        # LOBSTER formatına çevir
        lobster_data = self.convert_to_lobster_format(df)
        
        # LOBSTER formatında kaydet
        symbol = df['Symbol'].iloc[0] if 'Symbol' in df.columns else 'STOCK'
        output_dir = f"data/{symbol}"
        os.makedirs(output_dir, exist_ok=True)
        
        lobster_data.to_csv(f"{output_dir}/{symbol}_2025-07-01_2025-07-01_orderbook.csv", 
                           index=False, header=False)
        
        print(f"LOBSTER formatında veri kaydedildi: {output_dir}")
    
    def convert_to_lobster_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generic LOB verisini LOBSTER formatına çevirir
        
        Args:
            df (pd.DataFrame): LOB verisi
            
        Returns:
            pd.DataFrame: LOBSTER formatında veri
        """
        # LOBSTER formatı: [timestamp, bid_price_1, bid_volume_1, ask_price_1, ask_volume_1, ...]
        lobster_columns = ['timestamp']
        
        for level in range(1, 6):
            lobster_columns.extend([
                f'bid_price_{level}',
                f'bid_volume_{level}',
                f'ask_price_{level}',
                f'ask_volume_{level}'
            ])
        
        lobster_data = pd.DataFrame(columns=lobster_columns)
        
        # Timestamp'i Unix timestamp'e çevir
        lobster_data['timestamp'] = df['DateTime'].astype(np.int64) // 10**9
        
        # LOB verilerini ekle
        for level in range(1, 6):
            lobster_data[f'bid_price_{level}'] = df[f'Level {level} Bid Price']
            lobster_data[f'bid_volume_{level}'] = df[f'Level {level} Bid Volume']
            lobster_data[f'ask_price_{level}'] = df[f'Level {level} Ask Price']
            lobster_data[f'ask_volume_{level}'] = df[f'Level {level} Ask Volume']
        
        return lobster_data
    
    def get_feature_names(self) -> List[str]:
        """
        Özellik isimlerini döndürür
        
        Returns:
            List[str]: Özellik isimleri
        """
        feature_names = []
        
        for level in range(1, 6):
            feature_names.extend([
                f'Level_{level}_Bid_Price',
                f'Level_{level}_Bid_Volume',
                f'Level_{level}_Ask_Price',
                f'Level_{level}_Ask_Volume'
            ])
        
        return feature_names 