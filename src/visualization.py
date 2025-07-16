import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class LOBVisualizer:
    """
    Sadeleştirilmiş LOB veri görselleştirme sınıfı
    Ertesi gün fiyat tahmini odaklı
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_price_evolution(self, data_loader, start_idx: int = 0, end_idx: int = 1000):
        """
        Fiyat evrimini göster - sadece temel bilgi
        """
        data = data_loader.data.iloc[start_idx:end_idx]
        
        plt.figure(figsize=self.figsize)
        plt.plot(data.index, data['mid_price'], linewidth=1, alpha=0.8)
        plt.title(f'{data_loader.symbol} Fiyat Evrimi', fontsize=14, fontweight='bold')
        plt.xlabel('Zaman')
        plt.ylabel('Orta Fiyat')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_order_book_depth(self, data_loader, timestamp_idx: int = 0):
        """
        Belirli bir zamandaki order book derinliği
        """
        data = data_loader.data.iloc[timestamp_idx]
        
        # Bid ve ask fiyatları
        bid_prices = [data[f'bid_price_{i}'] for i in range(1, 6)]
        ask_prices = [data[f'ask_price_{i}'] for i in range(1, 6)]
        bid_volumes = [data[f'bid_volume_{i}'] for i in range(1, 6)]
        ask_volumes = [data[f'ask_volume_{i}'] for i in range(1, 6)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bid tarafı
        ax1.barh(range(5), bid_volumes, color='green', alpha=0.7)
        ax1.set_yticks(range(5))
        ax1.set_yticklabels([f'Bid {i}' for i in range(1, 6)])
        ax1.set_xlabel('Hacim')
        ax1.set_title('Bid Tarafı')
        ax1.grid(True, alpha=0.3)
        
        # Ask tarafı
        ax2.barh(range(5), ask_volumes, color='red', alpha=0.7)
        ax2.set_yticks(range(5))
        ax2.set_yticklabels([f'Ask {i}' for i in range(1, 6)])
        ax2.set_xlabel('Hacim')
        ax2.set_title('Ask Tarafı')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{data_loader.symbol} Order Book Derinliği', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, trainer):
        """
        Eğitim geçmişi - epoch bazında değişim grafikleri
        """
        history = trainer.history
        
        # Create subplots for detailed training analysis
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 15))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 1. Loss over epochs
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
        ax1.set_title('Model Loss Over Epochs', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(epochs)
        
        # 2. Accuracy over epochs
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=4)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
        ax2.set_title('Model Accuracy Over Epochs', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(epochs)
        
        # 3. Loss difference (overfitting indicator)
        loss_diff = [abs(train - val) for train, val in zip(history['train_loss'], history['val_loss'])]
        ax3.plot(epochs, loss_diff, 'g-', linewidth=2, marker='o', markersize=4)
        ax3.set_title('Train-Validation Loss Difference', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('|Train Loss - Val Loss|')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(epochs)
        
        # 4. Accuracy difference
        acc_diff = [abs(train - val) for train, val in zip(history['train_acc'], history['val_acc'])]
        ax4.plot(epochs, acc_diff, 'm-', linewidth=2, marker='o', markersize=4)
        ax4.set_title('Train-Validation Accuracy Difference', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('|Train Acc - Val Acc|')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(epochs)
        
        # 5. MAE over epochs
        if 'val_mae' in history:
            ax5.plot(epochs, history['val_mae'], 'orange', linewidth=2, marker='o', markersize=4, label='Validation MAE')
            ax5.set_title('Validation MAE Over Epochs', fontweight='bold', fontsize=12)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('MAE')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_xticks(epochs)
        
        # 6. MSE and MAPE over epochs
        if 'val_mse' in history and 'val_mape' in history:
            ax6_twin = ax6.twinx()
            
            line1 = ax6.plot(epochs, history['val_mse'], 'red', linewidth=2, marker='s', markersize=4, label='MSE')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('MSE', color='red')
            ax6.tick_params(axis='y', labelcolor='red')
            ax6.grid(True, alpha=0.3)
            ax6.set_xticks(epochs)
            
            line2 = ax6_twin.plot(epochs, history['val_mape'], 'purple', linewidth=2, marker='^', markersize=4, label='MAPE')
            ax6_twin.set_ylabel('MAPE (%)', color='purple')
            ax6_twin.tick_params(axis='y', labelcolor='purple')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax6.legend(lines, labels, loc='upper right')
            ax6.set_title('Validation MSE and MAPE Over Epochs', fontweight='bold', fontsize=12)
        
        plt.suptitle('Eğitim ve Validation Geçmişi - Epoch Bazında Değişim', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Additional summary statistics
        print(f"\n📊 Eğitim Özeti:")
        print(f"   Toplam Epoch: {len(epochs)}")
        print(f"   Son Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"   Son Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"   Son Train Acc: {history['train_acc'][-1]:.2f}%")
        print(f"   Son Val Acc: {history['val_acc'][-1]:.2f}%")
        print(f"   En İyi Val Loss: {min(history['val_loss']):.4f} (Epoch {history['val_loss'].index(min(history['val_loss'])) + 1})")
        print(f"   En İyi Val Acc: {max(history['val_acc']):.2f}% (Epoch {history['val_acc'].index(max(history['val_acc'])) + 1})")
        
        # Add metrics summary if available
        if 'val_mae' in history:
            print(f"\n📈 Validation Metrikleri:")
            print(f"   Son Val MAE: {history['val_mae'][-1]:.4f}")
            print(f"   Son Val MSE: {history['val_mse'][-1]:.4f}")
            print(f"   Son Val MAPE: {history['val_mape'][-1]:.2f}%")
            print(f"   Son Val Direction Acc: {history['val_direction_acc'][-1]:.2f}%")
            print(f"   En İyi Val MAE: {min(history['val_mae']):.4f} (Epoch {history['val_mae'].index(min(history['val_mae'])) + 1})")
            print(f"   En İyi Val MSE: {min(history['val_mse']):.4f} (Epoch {history['val_mse'].index(min(history['val_mse'])) + 1})")
            print(f"   En İyi Val MAPE: {min(history['val_mape']):.2f}% (Epoch {history['val_mape'].index(min(history['val_mape'])) + 1})")
            print(f"   En İyi Val Direction Acc: {max(history['val_direction_acc']):.2f}% (Epoch {history['val_direction_acc'].index(max(history['val_direction_acc'])) + 1})")
    
    def plot_prediction_results(self, results: Dict[str, Any], data_loader, start_idx: int = 0, end_idx: int = 500):
        """
        Tahmin sonuçları - sadece temel metrikler
        """
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Down', 'Stable', 'Up'],
                   yticklabels=['Down', 'Stable', 'Up'])
        plt.title('Confusion Matrix')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        plt.tight_layout()
        plt.show()
        
        # Sınıf dağılımı
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gerçek sınıflar
        y_true = results['true_labels']
        true_counts = pd.Series(y_true).value_counts().sort_index()
        ax1.bar(['Down', 'Stable', 'Up'], true_counts.values, color=['red', 'gray', 'green'], alpha=0.7)
        ax1.set_title('Gerçek Sınıf Dağılımı')
        ax1.set_ylabel('Sayı')
        
        # Tahmin sınıfları
        y_pred = results['predictions']
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        ax2.bar(['Down', 'Stable', 'Up'], pred_counts.values, color=['red', 'gray', 'green'], alpha=0.7)
        ax2.set_title('Tahmin Sınıf Dağılımı')
        ax2.set_ylabel('Sayı')
        
        plt.suptitle('Sınıf Dağılımları', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], top_n: int = 20):
        """
        Özellik önem sıralaması
        """
        # En önemli özellikleri al
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(features)), importance, color='skyblue', alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Önem Skoru')
        plt.title(f'En Önemli {top_n} Özellik')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_forecast(self, forecast):
        """
        N günlük mid price tahmin sonuçlarını bar/çizgi grafik olarak gösterir.
        Args:
            forecast: List of dicts with keys 'day', 'prediction', 'confidence', 'probabilities', 'mid_price', 'price_change_pct'
        """
        days = [f['day'] for f in forecast]
        preds = [f['prediction'] for f in forecast]
        confs = [f['confidence'] for f in forecast]
        mid_prices = [f['mid_price'] for f in forecast]
        class_names = ['Down', 'Stable', 'Up']
        colors = ['red', 'gray', 'green']
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Class predictions and confidence
        ax1.bar(days, preds, color=[colors[p] for p in preds], alpha=0.6, label='Tahmin Sınıfı')
        ax1.plot(days, confs, 'o-', color='blue', label='Güven Skoru', linewidth=2, markersize=8)
        ax1.set_xticks(days)
        ax1.set_xticklabels([f'Gün {d}' for d in days])
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(class_names)
        ax1.set_ylabel('Tahmin Sınıfı')
        ax1.set_title(f'{len(days)} Günlük Mid Price Yönü Tahmini ve Güven Skoru')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mid price evolution
        ax2.plot(days, mid_prices, 'o-', color='green', linewidth=3, markersize=10, label='Tahmini Mid Price')
        ax2.fill_between(days, mid_prices, alpha=0.3, color='lightgreen')
        ax2.set_xticks(days)
        ax2.set_xticklabels([f'Gün {d}' for d in days])
        ax2.set_ylabel('Mid Price')
        ax2.set_title(f'{len(days)} Günlük Mid Price Tahmini')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add mid price labels on the price plot
        for i, mid_price in enumerate(mid_prices):
            ax2.annotate(f'{mid_price:.4f}', (days[i], mid_price), 
                        textcoords="offset points", xytext=(0,10), 
                        ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_price_prediction_summary(self, data_loader, results: Optional[Dict[str, Any]] = None):
        """
        Fiyat tahmini özeti - sadece temel bilgiler
        """
        print(f"\n{'='*50}")
        print(f"{data_loader.symbol} FİYAT TAHMİNİ ÖZETİ")
        print(f"{'='*50}")
        
        print(f"Veri Seti Boyutu: {len(data_loader.data):,} kayıt")
        print(f"Tarih Aralığı: {data_loader.data.index[0]} - {data_loader.data.index[-1]}")
        print(f"Fiyat Aralığı: {data_loader.data['mid_price'].min():.4f} - {data_loader.data['mid_price'].max():.4f}")
        
        if results:
            print(f"\nModel Performansı:")
            print(f"Accuracy: {results['accuracy']:.3f}")
            
            # Classification report'dan metrikleri al
            report = results['classification_report']
            weighted_precision = report['weighted avg']['precision']
            weighted_recall = report['weighted avg']['recall']
            weighted_f1 = report['weighted avg']['f1-score']
            
            print(f"Precision: {weighted_precision:.3f}")
            print(f"Recall: {weighted_recall:.3f}")
            print(f"F1-Score: {weighted_f1:.3f}")
        
        print(f"{'='*50}\n")
    
    def plot_test_metrics(self, metrics: Dict[str, Any]):
        """
        Test metriklerini görselleştir - MAE, MSE, MAPE, Direction Accuracy
        Args:
            metrics: Test evaluation metrics dictionary
        """
        # Create a simple visualization of the 4 core metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Bar chart of MAE, MSE, MAPE, Direction Accuracy
        metric_names = ['MAE', 'MSE', 'MAPE (Error %)', 'Dir Acc (%)']
        metric_values = [metrics['mae'], metrics['mse'], metrics['mape'], metrics.get('direction_accuracy', 0)]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.8)
        ax1.set_title('Test Metrics: MAE, MSE, Classification Error, Direction Accuracy', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            if 'MAPE' in metric_names[metric_values.index(value)] or 'Dir Acc' in metric_names[metric_values.index(value)]:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
            else:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Pie chart showing metric proportions
        # Normalize values for better visualization
        normalized_values = [metrics['mae'], metrics['mse'], metrics['mape']/100, metrics.get('direction_accuracy', 0)/100]
        ax2.pie(normalized_values, labels=metric_names, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax2.set_title('Metric Distribution (Normalized)', fontweight='bold', fontsize=14)
        
        plt.suptitle('Test Evaluation Results - Core Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Test Evaluation Summary:")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   MSE: {metrics['mse']:.4f}")
        print(f"   MAPE (Classification Error): {metrics['mape']:.2f}%")
        print(f"   Direction Accuracy: {metrics.get('direction_accuracy', 0):.2f}%")
        
        # Print class distribution if available
        if 'class_distribution' in metrics:
            print(f"\n📊 Class Distribution:")
            print(f"   Predictions: {metrics['class_distribution']['predictions']}")
            print(f"   Targets: {metrics['class_distribution']['targets']}") 