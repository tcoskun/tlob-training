#!/usr/bin/env python3
"""
TLOB Fiyat Tahmin Projesi - Ana Çalıştırma Dosyası
"""

import os
import sys
import yaml
import warnings
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import sys
import os

# Proje dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import LOBDataLoader
from src.data_preprocessor import LOBDataPreprocessor
from src.tlob_integration import TLOBIntegration
from src.visualization import LOBVisualizer

def load_config():
    """Konfigürasyon dosyasını yükle"""
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def main():
    """Ana çalıştırma fonksiyonu"""
    print("🚀 TLOB (Time-weighted Limit Order Book) Analysis with Real TLOB Library")
    print("=" * 70)
    
    # Konfigürasyon yükle
    config = load_config()
    print(f"📋 Configuration loaded: {config['model']['type']} model")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Find data file
    data_source = config.get('data', {}).get('data_source', 'data/*.csv')
    if data_source.endswith('*.csv'):
        # Use glob pattern
        data_files = glob.glob(data_source)
        if not data_files:
            print("❌ No CSV files found in data directory!")
            return
        data_path = data_files[0]
    else:
        # Use specific file path
        data_path = data_source
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            return
    
    print(f"📊 Loading data from: {data_path}")
    
    # Load and preprocess data
    print("\n📈 Loading and preprocessing data...")
    data_loader = LOBDataLoader(data_path)
    df = data_loader.load_data()
    
    preprocessor = LOBDataPreprocessor()
    # For TLOB integration, we don't need to preprocess the data here
    # as it will be handled by the TLOB integration
    
    print(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Initialize TLOB integration
    print("\n🤖 Initializing TLOB model...")
    tlob_config = {
        'hidden_dim': config['model']['hidden_dim'],
        'num_layers': config['model']['num_layers'],
        'seq_size': config['model']['seq_size'],
        'num_heads': config['model'].get('num_heads', 1),
        'is_sin_emb': config['model'].get('is_sin_emb', True),
        'lr': config['training']['learning_rate'],
        'batch_size': config['training']['batch_size'],
        'horizon': config['model']['horizon'],
        'forecast_days': config['model']['forecast_days'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'min_delta': config['training'].get('min_delta', 0.001)
    }
    
    tlob_integration = TLOBIntegration(tlob_config)
    
    # Prepare data for TLOB
    print("🔄 Preparing data for TLOB model...")
    num_features = tlob_integration.prepare_data(data_path)
    print(f"✅ Data prepared with {num_features} features")
    
    # Create and train model
    print("\n🏋️ Training TLOB model...")
    model = tlob_integration.create_model(num_features)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train the model
    max_epochs = config['training']['epochs']
    training_history = tlob_integration.train_model(max_epochs=max_epochs)
    print("✅ Training completed!")
    
    # Load best model
    print("\n📥 Loading best model...")
    model.load_state_dict(torch.load('models/best_tlob_model.pth'))
    tlob_integration.model = model
    
    # Evaluate model on test data
    print("\n🧪 Evaluating model performance...")
    test_metrics = tlob_integration.evaluate_model()
    
    # Save test metrics
    tlob_integration.save_test_metrics(test_metrics)
    
    # Initialize visualizer
    viz = LOBVisualizer()
    
    # Visualize test metrics
    print("\n📊 Creating test metrics visualizations...")
    viz.plot_test_metrics(test_metrics)
    
    # Make predictions
    print("\n🔮 Making predictions...")
    
    # Get test data for predictions
    test_loader = tlob_integration.data_module.test_dataloader()
    test_batch = next(iter(test_loader))
    test_data = test_batch[0]  # Get first batch of data
    test_labels = test_batch[1]  # Get first batch of labels
    
    # Single prediction
    prediction, probabilities = tlob_integration.predict(test_data[0].numpy())
    print(f"📊 Single prediction: Class {prediction} with confidence {np.max(probabilities):.2%}")
    
    # N-day forecast
    forecast_days = config['model']['forecast_days']
    print(f"\n📅 Generating {forecast_days}-day forecast...")
    forecast = tlob_integration.predict_days(test_data.numpy())

    # Log forecast results
    print(f"\n{forecast_days} Günlük Mid Price Tahmin Sonuçları:")
    print("=" * 60)
    direction_map = {0: "Yükseliş", 1: "Sabit", 2: "Düşüş"}
    
    for day_result in forecast:
        day = day_result['day']
        pred = day_result['prediction']
        conf = day_result['confidence']
        mid_price = day_result['mid_price'].item() if hasattr(day_result['mid_price'], 'item') else float(day_result['mid_price'])
        change_pct = day_result['price_change_pct'].item() if hasattr(day_result['price_change_pct'], 'item') else float(day_result['price_change_pct'])
        direction = direction_map[pred]
        
        print(f"  Gün {day}: {direction} | Mid Price: {mid_price:.4f} | Değişim: {change_pct:+.2f}% | Güven: {conf:.2%}")
    
    print("=" * 60)

    # Visualize results
    print("\n📊 Creating visualizations...")
    
    # Plot training history
    print("📈 Plotting training history...")
    viz.plot_training_history(tlob_integration)
    
    # Plot other visualizations
    viz.plot_price_evolution(data_loader)
    viz.plot_forecast(forecast)
    
    print("\n🎉 Analysis completed successfully!")
    print("📁 Results saved in 'results/' directory")
    print("💾 Best model saved as 'models/best_tlob_model.pth'")

if __name__ == "__main__":
    main() 