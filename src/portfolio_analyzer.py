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
        Load LOB data and extract mid prices
        
        Args:
            data_dir: Directory containing LOB data files
            
        Returns:
            DataFrame with mid prices
        """
        print(f"📊 Loading LOB data from {data_dir} directory...")
        
        # Find CSV files in data directory
        data_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not data_files:
            raise ValueError(f"No CSV files found in {data_dir} directory")
            
        # Load first CSV file
        file_path = data_files[0]
        filename = os.path.basename(file_path)
        symbol = filename.split('-')[1] if '-' in filename else filename.split('.')[0]
        
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
        
        print(f"✅ Loaded {len(df)} price points for {symbol}")
        return df[['Mid_Price']].rename(columns={'Mid_Price': symbol})
    
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
            # Momentum strategy: buy when price increases, sell when decreases
            returns = prices.pct_change()
            decisions = np.where(returns > 0.001, 1,  # Buy on positive momentum
                               np.where(returns < -0.001, -1, 0))  # Sell on negative momentum
            
        elif strategy_type == 'mean_reversion':
            # Mean reversion strategy
            ma_short = prices.rolling(5).mean()
            ma_long = prices.rolling(20).mean()
            decisions = np.where(prices > ma_long * 1.01, -1,  # Sell when overbought
                               np.where(prices < ma_long * 0.99, 1, 0))  # Buy when oversold
            
        elif strategy_type == 'random':
            # Random strategy for testing
            np.random.seed(42)
            decisions = np.random.choice([-1, 0, 1], size=len(prices), p=[0.2, 0.6, 0.2])
            
        else:
            # Default: hold position
            decisions = np.zeros(len(prices))
        
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
        
        # Normalize weights (sum to 1 or 0)
        weights = decisions.div(decisions.abs().sum(axis=1), axis=0).fillna(0)
        
        # Create portfolio using from_orders
        portfolio = vbt.Portfolio.from_orders(
            close=price_data,
            size=weights,
            size_type='targetpercent',
            init_cash=init_cash,
            freq="1T",
            cash_sharing=True,
            call_seq='auto'
        )
        
        self.portfolio = portfolio
        self.decisions = decisions
        self.price_data = price_data
        
        print(f"✅ Portfolio created successfully using from_orders")
        return portfolio
    
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
        
        # Calculate annualization factor
        ann_factor = self.portfolio.returns().vbt.returns().ann_factor
        
        # Extract key metrics
        stats = {
            'total_return': full_stats['Total Return [%]'] / 100,
            'sharpe_ratio': full_stats['Sharpe Ratio'],
            'max_drawdown': full_stats['Max Drawdown [%]'] / 100,
            'annualized_return': (self.portfolio.returns().mean() * ann_factor),
            'annualized_volatility': self.portfolio.returns().std() * (ann_factor ** 0.5),
            'win_rate': 0.0  # Will calculate manually
        }
        
        # Calculate win rate manually
        try:
            returns = self.portfolio.returns()
            if len(returns) > 0:
                positive_returns = np.sum(returns > 0)
                stats['win_rate'] = positive_returns / len(returns)
        except Exception as e:
            print(f"⚠️ Error calculating win_rate: {e}")
        
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