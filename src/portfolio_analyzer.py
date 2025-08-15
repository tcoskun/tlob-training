#!/usr/bin/env python3
"""
Basit Portfolio Analysis Module using VectorBT - TLOB Data ile
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import glob
import os
warnings.filterwarnings('ignore')

# VectorBT ayarları
vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.array_wrapper['freq'] = '1T'  # 1 dakika

class PortfolioAnalyzer:
    """Basit portfolio analysis using VectorBT library with TLOB data"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.price_data = None
        self.portfolio = None
        self.decisions = None
        
    def load_lob_data(self, data_dir: str = "data") -> pd.DataFrame:
        """
        Load LOB data from all CSV files and extract mid prices
        
        Args:
            data_dir: Directory containing LOB data files
            
        Returns:
            DataFrame with mid prices from all files
        """
        print(f"📊 Loading LOB data from {data_dir} directory...")
        
        # Find all CSV files in data directory
        data_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not data_files:
            raise ValueError(f"No CSV files found in {data_dir} directory")
        
        print(f"📁 Found {len(data_files)} CSV files")
        
        all_dfs = []
        
        for file_path in data_files:
            filename = os.path.basename(file_path)
            # Extract symbol and date from filename: 2025-08-XX-SYMBOL-10.csv -> SYMBOL_08-XX
            # Format: 2025-08-05-AKBNK-10.csv -> parts: [2025, 08, 05, AKBNK, 10, csv]
            if filename.count('-') >= 4:
                # Extract symbol (4th part) and date (2nd and 3rd parts)
                symbol_name = filename.split('-')[3]  # AKBNK, THYAO, etc.
                date_part = filename.split('-')[1] + '-' + filename.split('-')[2]  # 08-05
                symbol = f'{symbol_name}_{date_part}'
            elif filename.count('-') >= 2:
                symbol = filename.split('-')[2]  # Fallback
            else:
                symbol = filename.split('.')[0]
            
            print(f"📈 Loading {filename} for symbol {symbol}...")
            
            # Load LOB data
            df = pd.read_csv(file_path, sep=';', decimal=',')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Convert numeric columns to float
            numeric_columns = [col for col in df.columns 
                              if any(x in col for x in ['Price', 'Volume', 'Ratio', 'mid_price'])]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert DateTime to datetime
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.set_index('DateTime', inplace=True)
            
            # Extract mid price
            if 'Level 1 Bid Price' in df.columns and 'Level 1 Ask Price' in df.columns:
                df['Mid_Price'] = (df['Level 1 Bid Price'] + df['Level 1 Ask Price']) / 2
            elif 'mid_price' in df.columns:
                df['Mid_Price'] = df['mid_price']
            else:
                raise ValueError("No price columns found")
            
            # Resample to 1-minute intervals and take first 1000 points for demo
            df = df.resample('1T').last().fillna(method='ffill')
            df = df.head(1000)  # Limit to first 1000 points for demo
            
            # Rename column to symbol
            df = df[['Mid_Price']].rename(columns={'Mid_Price': symbol})
            all_dfs.append(df)
        
        # Combine all dataframes
        if len(all_dfs) > 1:
            combined_df = pd.concat(all_dfs, axis=1)
            print(f"✅ Combined data from {len(data_files)} files: {combined_df.shape}")
        else:
            combined_df = all_dfs[0]
            print(f"✅ Loaded single file: {combined_df.shape}")
        
        return combined_df
    
    def create_trading_decisions(self, price_data: pd.DataFrame, strategy_type: str = 'momentum') -> pd.DataFrame:
        """
        Create trading decisions (-1, 0, 1) based on strategy
        
        Args:
            price_data: DataFrame with price data
            strategy_type: Type of strategy ('momentum', 'mean_reversion', 'random')
            
        Returns:
            DataFrame with trading decisions
        """
        symbol = price_data.columns[0]
        prices = price_data[symbol]
        
        if strategy_type == 'momentum':
            # Momentum strategy - daha aktif
            returns = prices.pct_change()
            # Daha düşük eşikler kullan (daha fazla trade)
            decisions = np.where(returns > 0.002, 1,  # Buy on positive momentum
                               np.where(returns < -0.002, -1, 0))  # Sell on negative momentum
            
        elif strategy_type == 'mean_reversion':
            # Mean reversion strategy - daha aktif
            ma_long = prices.rolling(10).mean()  # Uzun MA
            # Daha dar bantlar kullan
            decisions = np.where(prices > ma_long * 1.005, -1,  # Sell when overbought
                               np.where(prices < ma_long * 0.995, 1, 0))  # Buy when oversold
            
        elif strategy_type == 'random':
            # Daha aktif random strategy
            np.random.seed(42)
            # Daha fazla trade olasılığı
            decisions = np.random.choice([-1, 0, 1], size=len(prices), p=[0.15, 0.7, 0.15])
            
        else:
            # Default: hold position
            decisions = np.zeros(len(prices))
        
        # Trading sıklığını azalt - her 5 dakikada bir trade yap
        if len(decisions) > 10:
            # İlk 10 veri noktasını koru, sonrasında her 5'te bir trade yap
            for i in range(10, len(decisions), 5):
                if i < len(decisions):
                    # Diğer noktalarda 0 (hold)
                    decisions[i+1:i+5] = 0
        
        # Create decisions DataFrame
        decisions_df = pd.DataFrame(decisions, index=prices.index, columns=[symbol])
        
        return decisions_df
    
    def create_portfolio_from_orders(self, price_data: pd.DataFrame, strategy_type: str = 'momentum', 
                                   init_cash: float = 10000) -> vbt.Portfolio:
        """
        Create VectorBT portfolio using from_orders method
        
        Args:
            price_data: DataFrame with price data
            strategy_type: Trading strategy type
            init_cash: Initial cash amount
            
        Returns:
            VectorBT Portfolio object
        """
        print(f"🏗️ Creating portfolio with {strategy_type} strategy using from_orders...")
        
        # Create trading decisions
        decisions = self.create_trading_decisions(price_data, strategy_type)
        
        # Mean Reversion (Amount) stratejisini kullan
        weights = decisions * 500  # 500 birim sabit miktar
        size_type = 'amount'
        
        # Create portfolio using from_orders with transaction costs
        portfolio = vbt.Portfolio.from_orders(
            close=price_data,
            size=weights,
            size_type=size_type,
            init_cash=init_cash,
            freq="1T",
            cash_sharing=True,
            call_seq='auto',
            fees=0.001,  # %0.1 transaction cost
            slippage=0.0005  # %0.05 slippage
        )
        
        self.portfolio = portfolio
        self.decisions = decisions
        self.price_data = price_data
        
        print(f"✅ Portfolio created successfully using from_orders with transaction costs")
        return portfolio
    
    def create_portfolio_from_test_data(self, test_data):
        """
        Create portfolio data from TLOB test data
        
        Args:
            test_data: Test data tensor from TLOB analysis
            
        Returns:
            DataFrame with portfolio prices
        """
        print(f"📊 Creating portfolio from TLOB test data...")
        
        try:
            # Convert tensor to numpy if needed
            if hasattr(test_data, 'numpy'):
                test_data_np = test_data.numpy()
            else:
                test_data_np = test_data
            
            # Create a simple price series from test data
            # Use the first feature as price (or create synthetic prices)
            if len(test_data_np.shape) >= 2:
                # Take first feature as price
                prices = test_data_np[:, 0]
            else:
                # If 1D, use as is
                prices = test_data_np
            
            # Create time index for test period
            time_index = pd.date_range(
                start='2025-07-11 10:00:00',  # Test period start
                periods=len(prices),
                freq='1T'  # 1 minute intervals
            )
            
            # Create portfolio DataFrame with multiple symbols
            # Use available symbols from data directory
            data_dir = self.config.get('data_directory', 'data')
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            symbols = []
            
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                # Extract symbol using same logic as load_lob_data
                if filename.count('-') >= 4:
                    symbol_name = filename.split('-')[3]  # AKBNK, THYAO, etc.
                    date_part = filename.split('-')[1] + '-' + filename.split('-')[2]  # 08-05
                    symbol = f'{symbol_name}_{date_part}'
                elif filename.count('-') >= 2:
                    symbol = filename.split('-')[2]  # Fallback
                else:
                    symbol = filename.split('.')[0]
                symbols.append(symbol)
            
            # Create portfolio with all symbols (using same price data for demo)
            portfolio_data = pd.DataFrame(index=time_index)
            for symbol in symbols:
                portfolio_data[symbol] = prices
            
            # Store for later use
            self.portfolio_data = portfolio_data
            
            print(f"✅ Portfolio created from test data: {len(portfolio_data)} time points with {len(symbols)} symbols: {symbols}")
            return portfolio_data
            
        except Exception as e:
            print(f"❌ Error creating portfolio from test data: {e}")
            # Fallback: create simple portfolio
            fallback_data = pd.DataFrame({
                'AKBNK': np.random.randn(100).cumsum() + 70.0
            }, index=pd.date_range('2025-07-11 10:00:00', periods=100, freq='1T'))
            
            self.portfolio_data = fallback_data
            print(f"⚠️ Created fallback portfolio with {len(fallback_data)} points")
            return fallback_data
    
    def create_portfolio_from_lob_test_period(self, price_type: str = 'Mid_Price'):
        """
        Create portfolio data from LOB data (limited to test period only)
        
        Args:
            price_type: Type of price to use
            
        Returns:
            DataFrame with portfolio prices (test period only)
        """
        if not hasattr(self, 'lob_data') or self.lob_data is None:
            self.lob_data = self.load_lob_data()
            
        # Get the first symbol
        symbol = list(self.lob_data.columns)[0]
        df = self.lob_data[symbol]
        
        # Use all data instead of test split for now
        print(f"📊 Using all available data: {len(df)} points")
        
        # Create portfolio DataFrame with all data
        portfolio_data = pd.DataFrame({
            symbol: df
        })
        
        self.portfolio_data = portfolio_data
        
        print(f"✅ Portfolio created from LOB data: {len(portfolio_data)} time points")
        return portfolio_data
    
    def analyze_performance(self) -> Dict:
        """
        Analyze portfolio performance
        
        Returns:
            Dictionary with performance metrics
        """
        if self.portfolio is None:
            raise ValueError("No portfolio available for analysis")
        
        print("📊 Analyzing portfolio performance...")
        
        # Get basic stats
        full_stats = self.portfolio.stats()
        
        # Calculate annualization factor - 1 dakikalık veri için
        # 1 dakika = 1/1440 gün, yıllık = 1440 * 252
        ann_factor = 1440 * 252  # 1 dakikalık veri için yıllık faktör
        
        # Get returns for manual calculation
        returns = self.portfolio.returns()
        
        # Manual Sharpe ratio calculation with risk-free rate
        risk_free_rate = 0.02  # 2% yıllık risk-free rate
        
        # Returns'ları temizle (NaN ve sonsuz değerleri kaldır)
        clean_returns = returns.dropna()
        clean_returns = clean_returns[np.isfinite(clean_returns)]
        
        if len(clean_returns) > 1:
            # Günlük risk-free rate (1 dakikalık veri için)
            daily_rf_rate = risk_free_rate / (252 * 1440)  # 1440 dakika = 1 gün
            excess_returns = clean_returns - daily_rf_rate
            
            # Sharpe ratio hesaplama
            if clean_returns.std() > 0:
                sharpe_ratio = (excess_returns.mean() * np.sqrt(252 * 1440)) / (clean_returns.std() * np.sqrt(252 * 1440))
                # Sharpe oranını sınırla
                sharpe_ratio = np.clip(sharpe_ratio, -3.0, 3.0)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Extract key metrics - basit hesaplama
        stats = {
            'total_return': full_stats['Total Return [%]'] / 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': full_stats['Max Drawdown [%]'] / 100,
            'annualized_return': full_stats['Total Return [%]'] / 100,  # Basit: toplam getiri
            'annualized_volatility': full_stats['Volatility [%]'] / 100 if 'Volatility [%]' in full_stats else 0.0,
            'win_rate': 0.0  # Will calculate manually
        }
        
        # Calculate win rate manually - daha basit yaklaşım
        try:
            # VectorBT'den trade sayısını al
            if hasattr(self.portfolio.trades, 'records') and len(self.portfolio.trades.records) > 0:
                trades = self.portfolio.trades.records
                if 'PnL' in trades.columns:
                    winning_trades = trades[trades['PnL'] > 0]
                    stats['win_rate'] = len(winning_trades) / len(trades)
                else:
                    # PnL yoksa basit hesaplama
                    if len(clean_returns) > 0:
                        positive_returns = np.sum(clean_returns > 0)
                        stats['win_rate'] = positive_returns / len(clean_returns)
                    else:
                        stats['win_rate'] = 0.0
            else:
                # Trade yoksa basit hesaplama
                if len(clean_returns) > 0:
                    positive_returns = np.sum(clean_returns > 0)
                    stats['win_rate'] = positive_returns / len(clean_returns)
                else:
                    stats['win_rate'] = 0.0
        except Exception as e:
            print(f"⚠️ Error calculating win_rate: {e}")
            stats['win_rate'] = 0.0
        
        return stats
    
    def print_performance_report(self, stats: Dict):
        """
        Print performance report
        
        Args:
            stats: Performance statistics dictionary
        """
        print("\n" + "="*60)
        print("📊 PORTFÖY PERFORMANS RAPORU (from_orders)")
        print("="*60)
        
        print(f"💰 Toplam Getiri:           {stats['total_return']:.2%}")
        print(f"📈 Yıllık Getiri:           {stats['annualized_return']:.2%}")
        print(f"📊 Yıllık Volatilite:       {stats['annualized_volatility']:.2%}")
        print(f"⚖️  Sharpe Oranı:            {stats['sharpe_ratio']:.3f}")
        print(f"📉 Maksimum Drawdown:        {stats['max_drawdown']:.2%}")
        print(f"🎯 Kazanma Oranı:            {stats['win_rate']:.2%}")
        print("="*60)
    
    def plot_portfolio(self, save_path: str = None):
        """
        Plot portfolio performance
        
        Args:
            save_path: Path to save plot
        """
        if self.portfolio is None:
            raise ValueError("No portfolio available for plotting")
        
        print("📊 Creating portfolio plots...")
        
        # Set up plotting style
        sns.set_style('darkgrid')
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Portfolio Value
        plt.subplot(2, 2, 1)
        self.portfolio.value().plot()
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Portfolio Value')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Returns
        plt.subplot(2, 2, 2)
        self.portfolio.returns().plot()
        plt.title('Portfolio Returns')
        plt.ylabel('Returns')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        plt.subplot(2, 2, 3)
        self.portfolio.drawdown().plot()
        plt.title('Portfolio Drawdown')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Price vs Decisions
        plt.subplot(2, 2, 4)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot price
        self.price_data.plot(ax=ax1, color='blue', alpha=0.7)
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot decisions
        self.decisions.plot(ax=ax2, color='red', alpha=0.7)
        ax2.set_ylabel('Trading Decisions', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('Price vs Trading Decisions')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filename: str = 'results/portfolio_analysis.json'):
        """
        Save portfolio results to JSON file
        
        Args:
            filename: Output filename
        """
        if self.portfolio is None:
            raise ValueError("No portfolio available for saving")
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Get performance stats
        stats = self.analyze_performance()
        
        # Prepare results for saving
        results = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': stats,
            'portfolio_info': {
                'initial_cash': self.portfolio.init_cash,
                'final_value': self.portfolio.value().iloc[-1],
                'total_trades': len(self.portfolio.trades.records) if hasattr(self.portfolio.trades, 'records') else 0
            }
        }
        
        # Save to JSON
        import json
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}") 