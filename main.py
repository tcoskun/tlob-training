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
    
    # Find data file
    data_source = config.get('data', {}).get('data_source', 'data/*.csv')
    if data_source.endswith('*.csv'):
        # Use glob pattern
        data_files = glob.glob(data_source)
        if not data_files:
            print("❌ No CSV files found in data directory!")
            return None
        data_path = data_files[0]
    else:
        # Use specific file path
        data_path = data_source
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            return None
    
    print(f"📊 Loading data from: {data_path}")
    
    # Load and preprocess data
    print("\n📈 Loading and preprocessing data...")
    data_loader = LOBDataLoader(data_path)
    df = data_loader.load_data()
    
    preprocessor = LOBDataPreprocessor()
    
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
    
    return tlob_integration, test_metrics, forecast

def run_portfolio_analysis(config):
    """Portföy analizi çalıştır"""
    print("\n📊 PORTFÖY ANALİZİ (LOB Verileri ile)")
    print("=" * 50)
    
    # Portfolio configuration
    portfolio_config = config.get('portfolio', {})
    data_directory = portfolio_config.get('data_directory', 'data')
    price_type = portfolio_config.get('price_type', 'Mid_Price')
    weights = portfolio_config.get('weights', None)
    rebalance_freq = portfolio_config.get('rebalance_freq', 'H')
    
    print(f"📈 Portfolio Analysis Configuration:")
    print(f"   Data Directory: {data_directory}")
    print(f"   Price Type: {price_type}")
    print(f"   Rebalancing: {rebalance_freq}")
    print(f"   Weights: {weights if weights else 'Equal Weight'}")
    
    # Initialize portfolio analyzer
    analyzer = PortfolioAnalyzer(portfolio_config)
    
    # Load LOB data
    print("\n📊 Loading LOB data...")
    lob_data = analyzer.load_lob_data(data_directory)
    
    # Create portfolio from LOB data
    print("\n🏗️ Creating portfolio from LOB data...")
    portfolio_data = analyzer.create_portfolio_from_lob(price_type=price_type)
    
    # Calculate returns
    print("\n📈 Calculating returns...")
    returns = analyzer.calculate_returns()
    
    # Create portfolio strategy
    print("\n🏗️ Creating portfolio strategy...")
    portfolio = analyzer.create_portfolio_strategy(weights, rebalance_freq)
    
    # Analyze performance
    print("\n📊 Analyzing portfolio performance...")
    try:
        stats = analyzer.analyze_portfolio_performance()
        
        # Generate and print report
        print("\n" + analyzer.generate_portfolio_report())
        
        # Create visualizations
        print("\n📊 Creating portfolio visualizations...")
        try:
            analyzer.plot_portfolio_analysis('results/portfolio_analysis.png')
        except Exception as e:
            print(f"⚠️ Error creating portfolio visualizations: {e}")
        
        # Save results
        analyzer.save_portfolio_results()
        
    except Exception as e:
        print(f"❌ Error in portfolio analysis: {e}")
        stats = {}
    
    # Backtest different LOB-based strategies
    print("\n🔄 Backtesting LOB-based trading strategies...")
    strategies = portfolio_config.get('strategies', [])
    
    strategy_results = {}
    for strategy_config in strategies:
        strategy_name = strategy_config['name']
        strategy_params = strategy_config.get('params', {})
        
        print(f"\n📊 Backtesting {strategy_name} strategy...")
        try:
            backtest_portfolio = analyzer.backtest_lob_strategy(strategy_name, strategy_params)
            
            # Calculate strategy performance with error handling
            strategy_stats = {}
            
            def safe_extract_value(value):
                """Safely extract value from VectorBT Series or scalar"""
                try:
                    if hasattr(value, 'iloc'):
                        # It's a Series, get the first value
                        return float(value.iloc[0]) if len(value) > 0 else 0.0
                    elif hasattr(value, 'item'):
                        # It's a numpy scalar
                        return float(value.item())
                    else:
                        # It's a regular scalar
                        return float(value)
                except:
                    return 0.0
            
            try:
                strategy_stats['total_return'] = safe_extract_value(backtest_portfolio.total_return())
            except Exception as e:
                print(f"⚠️ Error calculating total_return for {strategy_name}: {e}")
                strategy_stats['total_return'] = 0.0
                
            try:
                strategy_stats['sharpe_ratio'] = safe_extract_value(backtest_portfolio.sharpe_ratio())
            except Exception as e:
                print(f"⚠️ Error calculating sharpe_ratio for {strategy_name}: {e}")
                strategy_stats['sharpe_ratio'] = 0.0
                
            try:
                strategy_stats['max_drawdown'] = safe_extract_value(backtest_portfolio.max_drawdown())
            except Exception as e:
                print(f"⚠️ Error calculating max_drawdown for {strategy_name}: {e}")
                strategy_stats['max_drawdown'] = 0.0
                
            # Calculate win rate manually since it's not available in VectorBT
            try:
                # Get trade statistics
                trades = backtest_portfolio.trades
                if hasattr(trades, 'records') and len(trades.records) > 0:
                    # Check what columns are available in trade records
                    available_columns = trades.records.columns.tolist()
                    print(f"📊 Available trade record columns for {strategy_name}: {available_columns}")
                    
                    # Try different possible column names for profit/loss
                    pnl_column = None
                    for col in ['PnL', 'pnl', 'profit', 'Profit', 'return', 'Return']:
                        if col in available_columns:
                            pnl_column = col
                            break
                    
                    if pnl_column:
                        # Calculate win rate from trade records
                        winning_trades = trades.records[trades.records[pnl_column] > 0]
                        total_trades = len(trades.records)
                        if total_trades > 0:
                            strategy_stats['win_rate'] = len(winning_trades) / total_trades
                        else:
                            strategy_stats['win_rate'] = 0.0
                    else:
                        # If no PnL column found, use returns-based calculation
                        print(f"⚠️ No PnL column found for {strategy_name}, using returns-based calculation")
                        strategy_stats['win_rate'] = 0.0
                else:
                    # Alternative: calculate from returns
                    returns = backtest_portfolio.returns()
                    if hasattr(returns, 'values'):
                        returns_values = returns.values
                    else:
                        returns_values = returns
                    
                    if len(returns_values) > 0:
                        positive_returns = np.sum(returns_values > 0)
                        strategy_stats['win_rate'] = positive_returns / len(returns_values)
                    else:
                        strategy_stats['win_rate'] = 0.0
            except Exception as e:
                print(f"⚠️ Error calculating win_rate for {strategy_name}: {e}")
                strategy_stats['win_rate'] = 0.0
            
            strategy_results[strategy_name] = strategy_stats
            
            print(f"   {strategy_name} Results:")
            print(f"     Total Return: {strategy_stats.get('total_return', 0):.2%}")
            print(f"     Sharpe Ratio: {strategy_stats.get('sharpe_ratio', 0):.2f}")
            print(f"     Max Drawdown: {strategy_stats.get('max_drawdown', 0):.2%}")
            print(f"     Win Rate: {strategy_stats.get('win_rate', 0):.2%}")
            
        except Exception as e:
            print(f"❌ Error backtesting {strategy_name}: {e}")
            strategy_results[strategy_name] = {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
    
    return analyzer, portfolio, strategy_results

def main():
    """Ana çalıştırma fonksiyonu"""
    print("🚀 TLOB (Time-weighted Limit Order Book) Analysis with Portfolio Analysis")
    print("=" * 80)
    
    # Konfigürasyon yükle
    config = load_config()
    print(f"📋 Configuration loaded")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run TLOB analysis
    tlob_results = run_tlob_analysis(config)
    
    # Run Portfolio analysis
    portfolio_results = run_portfolio_analysis(config)
    
    print("\n🎉 Complete analysis finished successfully!")
    print("📁 Results saved in 'results/' directory")
    print("💾 Best model saved as 'models/best_tlob_model.pth'")
    print("📊 Portfolio analysis saved as 'results/portfolio_analysis.json'")

if __name__ == "__main__":
    main() 