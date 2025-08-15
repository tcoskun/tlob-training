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
from src.portfolio_analyzer import PortfolioAnalyzer

def load_config():
    """Konfigürasyon dosyasını yükle"""
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def run_tlob_analysis(config):
    """TLOB analizi çalıştır"""
    print("\n🤖 TLOB MODEL ANALİZİ")
    print("=" * 50)
    
    # Find all data files
    data_source = config.get('data', {}).get('data_source', 'data/*.csv')
    
    if data_source.endswith('*.csv'):
        # Use glob pattern to find all CSV files
        data_files = glob.glob(data_source)
        if not data_files:
            print("❌ No CSV files found in data directory!")
            return None
        print(f"📊 Found {len(data_files)} CSV files: {[os.path.basename(f) for f in data_files]}")
        data_paths = data_files
    else:
        # Use specific file path
        data_paths = [data_source]
        if not os.path.exists(data_source):
            print(f"❌ Data file not found: {data_source}")
            return None
    
    print(f"📊 Loading data from {len(data_paths)} files...")
    
    # Load and preprocess data from all files
    print("\n📈 Loading and preprocessing data from all files...")
    all_dfs = []
    for data_path in data_paths:
        print(f"   📁 Loading: {os.path.basename(data_path)}")
        data_loader = LOBDataLoader(data_path)
        df = data_loader.load_data()
        all_dfs.append(df)
    
    # Combine all dataframes
    if len(all_dfs) > 1:
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"✅ Combined data from {len(data_paths)} files")
    else:
        df = all_dfs[0]
        print(f"✅ Loaded single file")
    
    preprocessor = LOBDataPreprocessor()
    
    print(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Initialize TLOB integration
    print("\n🤖 Initializing TLOB model...")
    print(f"📋 Config device setting: {config['training'].get('device', 'cpu')}")
    
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
        'min_delta': config['training'].get('min_delta', 0.001),
        'device': config['training'].get('device', 'cpu')  # Device ayarını ekle
    }
    
    print(f"🔧 TLOB config device: {tlob_config['device']}")
    
    tlob_integration = TLOBIntegration(tlob_config)
    
    # Device bilgisini göster
    print(f"🖥️ Using device: {tlob_integration.device}")
    
    # Prepare data for TLOB
    print("🔄 Preparing data for TLOB model...")
    num_features = tlob_integration.prepare_data(data_paths)
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
    
    return tlob_integration, test_metrics, forecast

def run_portfolio_analysis(config, test_data=None):
    """Portföy analizi çalıştır - sadece test seti üzerinde"""
    print("\n📊 PORTFÖY ANALİZİ (Test Seti Üzerinde)")
    print("=" * 50)
    
    # Portfolio configuration
    portfolio_config = config.get('portfolio', {})
    data_directory = portfolio_config.get('data_directory', 'data')
    strategy_type = portfolio_config.get('strategy_type', 'mean_reversion')
    init_cash = portfolio_config.get('init_cash', 10000)
    
    print(f"📈 Portfolio Analysis Configuration:")
    print(f"   Data Directory: {data_directory}")
    print(f"   Strategy Type: {strategy_type}")
    print(f"   Initial Cash: {init_cash}")
    print(f"   Analysis Scope: Test Set Only")
    
    # Initialize portfolio analyzer
    analyzer = PortfolioAnalyzer(portfolio_config)
    
    if test_data is not None:
        # Use test data from TLOB analysis
        print("\n📊 Using test data from TLOB analysis...")
        print(f"   Test data shape: {test_data.shape}")
        
        # Create portfolio from test data
        print("\n🏗️ Creating portfolio from test data...")
        portfolio_data = analyzer.create_portfolio_from_test_data(test_data)
        
    else:
        # Fallback: Load LOB data but limit to test period
        print("\n📊 Loading LOB data (limited to test period)...")
        lob_data = analyzer.load_lob_data(data_directory)
        
        # Create portfolio from LOB data (limited to test period)
        print("\n🏗️ Creating portfolio from LOB data (test period only)...")
        portfolio_data = analyzer.create_portfolio_from_lob_test_period(price_type='Mid_Price')
    
    # Create portfolio using from_orders method
    print("\n🏗️ Creating portfolio strategy...")
    portfolio = analyzer.create_portfolio_from_orders(portfolio_data, strategy_type, init_cash)
    
    # Analyze performance
    print("\n📊 Analyzing portfolio performance...")
    try:
        stats = analyzer.analyze_performance()
        
        # Print performance report
        analyzer.print_performance_report(stats)
        
        # Create visualizations
        print("\n📊 Creating portfolio visualizations...")
        try:
            analyzer.plot_portfolio('results/portfolio_analysis_test.png')
        except Exception as e:
            print(f"⚠️ Error creating portfolio visualizations: {e}")
        
        # Save results
        analyzer.save_results()
        
    except Exception as e:
        print(f"❌ Error in portfolio analysis: {e}")
        stats = {}
    
    return analyzer, portfolio, stats

def main():
    """Ana çalıştırma fonksiyonu"""
    print("🚀 TLOB Model + Portfolio Analysis")
    print("=" * 80)
    
    # Konfigürasyon yükle
    config = load_config()
    print(f"📋 Configuration loaded")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run TLOB analysis first
    print("\n" + "="*60)
    print("🤖 TLOB ANALYSIS")
    print("="*60)
    tlob_results = run_tlob_analysis(config)
    
    # Extract test data from TLOB results
    test_data = None
    if tlob_results and len(tlob_results) > 0:
        tlob_integration = tlob_results[0]
        if hasattr(tlob_integration, 'data_module') and tlob_integration.data_module:
            # Get test data from TLOB data module
            try:
                test_loader = tlob_integration.data_module.test_dataloader()
                test_batch = next(iter(test_loader))
                test_data = test_batch[0]  # Get test features
                print(f"📊 Extracted test data from TLOB: {test_data.shape}")
            except Exception as e:
                print(f"⚠️ Could not extract test data from TLOB: {e}")
    
    # Run Portfolio analysis
    print("\n" + "="*60)
    print("📊 PORTFOLIO ANALYSIS")
    print("="*60)
    portfolio_results = run_portfolio_analysis(config, test_data)
    
    print("\n🎉 Complete analysis finished successfully!")
    print("📁 Results saved in 'results/' directory")
    print("💾 Best model saved as 'models/best_tlob_model.pth'")
    print("📊 Portfolio analysis saved as 'results/portfolio_analysis_test.png'")

if __name__ == "__main__":
    main() 