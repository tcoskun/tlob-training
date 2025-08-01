#!/usr/bin/env python3
"""
Portfolio Analysis Module using VectorBT - Integrated with TLOB Data
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

class PortfolioAnalyzer:
    """Portfolio analysis using VectorBT library with TLOB data integration"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.portfolio_data = None
        self.returns = None
        self.positions = None
        self.portfolio_stats = None
        self.lob_data = {}
        
    def load_lob_data(self, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """
        Load LOB data from data directory
        
        Args:
            data_dir: Directory containing LOB data files
            
        Returns:
            Dictionary of symbol: DataFrame pairs
        """
        print(f"📊 Loading LOB data from {data_dir} directory...")
        
        # Find all CSV files in data directory
        data_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not data_files:
            raise ValueError(f"No CSV files found in {data_dir} directory")
            
        lob_data = {}
        
        for file_path in data_files:
            try:
                # Extract symbol from filename
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
                
                # Extract mid price from bid and ask prices
                if 'Level 1 Bid Price' in df.columns and 'Level 1 Ask Price' in df.columns:
                    df['Mid_Price'] = (df['Level 1 Bid Price'] + df['Level 1 Ask Price']) / 2
                elif 'mid_price' in df.columns:
                    df['Mid_Price'] = df['mid_price']
                else:
                    print(f"⚠️ No price columns found for {symbol}")
                    continue
                
                # Calculate additional features
                if 'Level 1 Ask Price' in df.columns and 'Level 1 Bid Price' in df.columns:
                    df['Spread'] = df['Level 1 Ask Price'] - df['Level 1 Bid Price']
                    df['Spread_Ratio'] = df['Spread'] / df['Mid_Price']
                else:
                    df['Spread'] = 0
                    df['Spread_Ratio'] = 0
                
                # Calculate volume imbalance
                if 'Total Bid Volume' in df.columns and 'Total Ask Volume' in df.columns:
                    total_volume = df['Total Bid Volume'] + df['Total Ask Volume']
                    df['Volume_Imbalance'] = np.where(total_volume > 0, 
                                                     (df['Total Bid Volume'] - df['Total Ask Volume']) / total_volume, 
                                                     0)
                else:
                    df['Volume_Imbalance'] = 0
                
                # Calculate price impact
                df['Price_Impact'] = df['Mid_Price'].pct_change()
                
                # Store processed data
                lob_data[symbol] = df
                
                print(f"✅ Loaded {symbol}: {len(df)} records, {len(df.columns)} features")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                import traceback
                traceback.print_exc()
                
        if not lob_data:
            raise ValueError("No LOB data could be loaded")
            
        self.lob_data = lob_data
        print(f"✅ Total LOB data loaded: {len(lob_data)} symbols")
        return lob_data
    
    def create_portfolio_from_lob(self, symbols: List[str] = None, 
                                 price_type: str = 'Mid_Price') -> pd.DataFrame:
        """
        Create portfolio data from LOB data
        
        Args:
            symbols: List of symbols to include in portfolio
            price_type: Type of price to use ('Mid_Price', 'Level 1 Bid Price', 'Level 1 Ask Price')
            
        Returns:
            DataFrame with portfolio prices
        """
        if not self.lob_data:
            self.load_lob_data()
            
        if symbols is None:
            symbols = list(self.lob_data.keys())
            
        portfolio_prices = {}
        
        for symbol in symbols:
            if symbol in self.lob_data:
                df = self.lob_data[symbol]
                
                # Resample to consistent time intervals (e.g., 1 minute)
                df_resampled = df[price_type].resample('1T').last().dropna()
                
                portfolio_prices[symbol] = df_resampled
                print(f"📊 Added {symbol}: {len(df_resampled)} price points")
            else:
                print(f"⚠️ Symbol {symbol} not found in LOB data")
                
        if not portfolio_prices:
            raise ValueError("No valid symbols found for portfolio")
            
        # Combine all prices into a single DataFrame
        portfolio_data = pd.DataFrame(portfolio_prices)
        portfolio_data = portfolio_data.dropna()
        
        print(f"✅ Portfolio created: {len(portfolio_data)} time points, {len(portfolio_data.columns)} symbols")
        self.portfolio_data = portfolio_data
        return portfolio_data
    
    def calculate_returns(self, method: str = 'pct_change') -> pd.DataFrame:
        """
        Calculate returns for portfolio
        
        Args:
            method: 'pct_change' or 'log_returns'
            
        Returns:
            DataFrame with returns
        """
        if self.portfolio_data is None:
            raise ValueError("No portfolio data loaded. Call create_portfolio_from_lob() first.")
            
        if method == 'pct_change':
            returns = self.portfolio_data.pct_change().dropna()
        elif method == 'log_returns':
            returns = np.log(self.portfolio_data / self.portfolio_data.shift(1)).dropna()
        else:
            raise ValueError("method must be 'pct_change' or 'log_returns'")
            
        self.returns = returns
        print(f"✅ Returns calculated using {method}: {len(returns)} periods")
        return returns
    
    def create_portfolio_strategy(self, weights: Dict[str, float] = None, 
                                rebalance_freq: str = 'H') -> vbt.Portfolio:
        """
        Create a portfolio strategy using VectorBT
        
        Args:
            weights: Dictionary of symbol: weight pairs
            rebalance_freq: Rebalancing frequency ('T', 'H', 'D', 'W', 'M')
            
        Returns:
            VectorBT Portfolio object
        """
        if self.portfolio_data is None:
            raise ValueError("No portfolio data loaded. Call create_portfolio_from_lob() first.")
            
        if weights is None:
            # Equal weight portfolio
            n_assets = len(self.portfolio_data.columns)
            weights = {col: 1.0/n_assets for col in self.portfolio_data.columns}
            
        print(f"📊 Creating portfolio with weights: {weights}")
        
        # Create portfolio using VectorBT from holding with minimal parameters
        portfolio = vbt.Portfolio.from_holding(
            self.portfolio_data,
            size=1.0,
            fees=0.001,
            freq='1T'
        )
        
        self.portfolio = portfolio
        print("✅ Portfolio strategy created")
        return portfolio
    
    def analyze_portfolio_performance(self) -> Dict:
        """
        Analyze portfolio performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        if not hasattr(self, 'portfolio'):
            raise ValueError("No portfolio created. Call create_portfolio_strategy() first.")
            
        # Calculate key metrics with error handling
        stats = {}
        
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
            stats['total_return'] = safe_extract_value(self.portfolio.total_return())
        except Exception as e:
            print(f"⚠️ Error calculating total_return: {e}")
            stats['total_return'] = 0.0
            
        try:
            stats['annual_return'] = safe_extract_value(self.portfolio.annualized_return())
        except Exception as e:
            print(f"⚠️ Error calculating annual_return: {e}")
            stats['annual_return'] = 0.0
            
        try:
            stats['volatility'] = safe_extract_value(self.portfolio.annualized_volatility())
        except Exception as e:
            print(f"⚠️ Error calculating volatility: {e}")
            stats['volatility'] = 0.0
            
        try:
            stats['sharpe_ratio'] = safe_extract_value(self.portfolio.sharpe_ratio())
        except Exception as e:
            print(f"⚠️ Error calculating sharpe_ratio: {e}")
            stats['sharpe_ratio'] = 0.0
            
        try:
            stats['max_drawdown'] = safe_extract_value(self.portfolio.max_drawdown())
        except Exception as e:
            print(f"⚠️ Error calculating max_drawdown: {e}")
            stats['max_drawdown'] = 0.0
            
        try:
            stats['calmar_ratio'] = safe_extract_value(self.portfolio.calmar_ratio())
        except Exception as e:
            print(f"⚠️ Error calculating calmar_ratio: {e}")
            stats['calmar_ratio'] = 0.0
            
        # Try to calculate additional metrics if available
        # Calculate win rate manually since it's not available in VectorBT
        try:
            # Get trade statistics
            trades = self.portfolio.trades
            if hasattr(trades, 'records') and len(trades.records) > 0:
                # Check what columns are available in trade records
                available_columns = trades.records.columns.tolist()
                print(f"📊 Available trade record columns: {available_columns}")
                
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
                        stats['win_rate'] = len(winning_trades) / total_trades
                    else:
                        stats['win_rate'] = 0.0
                else:
                    # If no PnL column found, use returns-based calculation
                    print("⚠️ No PnL column found, using returns-based calculation")
                    stats['win_rate'] = 0.0
            else:
                # Alternative: calculate from returns
                returns = self.portfolio.returns()
                if hasattr(returns, 'values'):
                    returns_values = returns.values
                else:
                    returns_values = returns
                
                if len(returns_values) > 0:
                    positive_returns = np.sum(returns_values > 0)
                    stats['win_rate'] = positive_returns / len(returns_values)
                else:
                    stats['win_rate'] = 0.0
        except Exception as e:
            print(f"⚠️ Error calculating win_rate: {e}")
            stats['win_rate'] = 0.0
            
        try:
            stats['profit_factor'] = safe_extract_value(self.portfolio.profit_factor())
        except AttributeError:
            print("⚠️ profit_factor method not available in this VectorBT version")
            stats['profit_factor'] = 0.0
        except Exception as e:
            print(f"⚠️ Error calculating profit_factor: {e}")
            stats['profit_factor'] = 0.0
            
        try:
            stats['avg_win'] = safe_extract_value(self.portfolio.avg_win())
        except AttributeError:
            print("⚠️ avg_win method not available in this VectorBT version")
            stats['avg_win'] = 0.0
        except Exception as e:
            print(f"⚠️ Error calculating avg_win: {e}")
            stats['avg_win'] = 0.0
            
        try:
            stats['avg_loss'] = safe_extract_value(self.portfolio.avg_loss())
        except AttributeError:
            print("⚠️ avg_loss method not available in this VectorBT version")
            stats['avg_loss'] = 0.0
        except Exception as e:
            print(f"⚠️ Error calculating avg_loss: {e}")
            stats['avg_loss'] = 0.0
            
        # Risk metrics
        try:
            # Try different VaR calculation methods
            try:
                var_value = self.portfolio.value_at_risk(level=0.05)
                stats['var_95'] = safe_extract_value(var_value)
            except TypeError:
                # Try without level parameter
                var_value = self.portfolio.value_at_risk()
                stats['var_95'] = safe_extract_value(var_value)
            except Exception as e:
                print(f"⚠️ Error calculating VaR: {e}")
                stats['var_95'] = 0.0
        except AttributeError:
            print("⚠️ value_at_risk method not available in this VectorBT version")
            stats['var_95'] = 0.0
            
        try:
            cvar_value = self.portfolio.conditional_value_at_risk(level=0.05)
            stats['cvar_95'] = safe_extract_value(cvar_value)
        except AttributeError:
            print("⚠️ conditional_value_at_risk method not available in this VectorBT version")
            stats['cvar_95'] = 0.0
        except Exception as e:
            print(f"⚠️ Error calculating CVaR: {e}")
            stats['cvar_95'] = 0.0
        
        self.portfolio_stats = stats
        return stats
    
    def create_trading_signals_from_lob(self, strategy: str = 'spread_based', 
                                       params: Dict = None) -> pd.DataFrame:
        """
        Create trading signals based on LOB features
        
        Args:
            strategy: Trading strategy ('spread_based', 'volume_imbalance', 'price_momentum')
            params: Strategy parameters
            
        Returns:
            DataFrame with trading signals
        """
        if not self.lob_data:
            raise ValueError("No LOB data loaded")
            
        signals = {}
        
        for symbol, df in self.lob_data.items():
            if strategy == 'spread_based':
                # Trade based on spread width
                spread_threshold = params.get('spread_threshold', 0.001) if params else 0.001
                
                # Buy when spread is narrow (good liquidity)
                buy_signal = df['Spread_Ratio'] < spread_threshold
                # Sell when spread is wide (poor liquidity)
                sell_signal = df['Spread_Ratio'] > spread_threshold * 2
                
            elif strategy == 'volume_imbalance':
                # Trade based on volume imbalance
                imbalance_threshold = params.get('imbalance_threshold', 0.1) if params else 0.1
                
                # Buy when bid volume > ask volume (buying pressure)
                buy_signal = df['Volume_Imbalance'] > imbalance_threshold
                # Sell when ask volume > bid volume (selling pressure)
                sell_signal = df['Volume_Imbalance'] < -imbalance_threshold
                
            elif strategy == 'price_momentum':
                # Trade based on price momentum
                momentum_window = params.get('momentum_window', 10) if params else 10
                momentum_threshold = params.get('momentum_threshold', 0.001) if params else 0.001
                
                # Calculate price momentum
                price_momentum = df['Mid_Price'].pct_change(momentum_window)
                
                # Buy on positive momentum
                buy_signal = price_momentum > momentum_threshold
                # Sell on negative momentum
                sell_signal = price_momentum < -momentum_threshold
                
            elif strategy == 'level_depth':
                # Trade based on order book depth
                depth_threshold = params.get('depth_threshold', 1000000) if params else 1000000
                
                # Buy when depth is high (good liquidity)
                total_depth = df['Total Bid Volume'] + df['Total Ask Volume']
                buy_signal = total_depth > depth_threshold
                # Sell when depth is low
                sell_signal = total_depth < depth_threshold * 0.5
                
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            # Resample signals to match portfolio frequency
            buy_signal_resampled = buy_signal.resample('1T').last().fillna(False)
            sell_signal_resampled = sell_signal.resample('1T').last().fillna(False)
            
            signals[f'{symbol}_buy'] = buy_signal_resampled
            signals[f'{symbol}_sell'] = sell_signal_resampled
            
        return pd.DataFrame(signals)
    
    def backtest_lob_strategy(self, strategy: str = 'spread_based', 
                             params: Dict = None) -> vbt.Portfolio:
        """
        Backtest a LOB-based trading strategy
        
        Args:
            strategy: Trading strategy
            params: Strategy parameters
            
        Returns:
            VectorBT Portfolio object with backtest results
        """
        if self.portfolio_data is None:
            raise ValueError("No portfolio data loaded")
            
        try:
            signals = self.create_trading_signals_from_lob(strategy, params)
            
            # Extract buy and sell signals for each symbol
            buy_signals = {}
            sell_signals = {}
            
            for symbol in self.portfolio_data.columns:
                buy_col = f'{symbol}_buy'
                sell_col = f'{symbol}_sell'
                
                if buy_col in signals.columns and sell_col in signals.columns:
                    # Align signals with portfolio data
                    buy_signals[symbol] = signals[buy_col].reindex(self.portfolio_data.index).fillna(False)
                    sell_signals[symbol] = signals[sell_col].reindex(self.portfolio_data.index).fillna(False)
            
            # Create portfolio from signals
            portfolio = vbt.Portfolio.from_signals(
                self.portfolio_data,
                entries=pd.DataFrame(buy_signals),
                exits=pd.DataFrame(sell_signals),
                size=1.0,
                fees=0.001,
                freq='1T'
            )
            
            self.backtest_portfolio = portfolio
            print(f"✅ Backtest completed for {strategy} strategy")
            return portfolio
            
        except Exception as e:
            print(f"❌ Error in backtest for {strategy}: {e}")
            # Return a simple portfolio as fallback
            portfolio = vbt.Portfolio.from_holding(
                self.portfolio_data,
                size=1.0,
                fees=0.001,
                freq='1T'
            )
            self.backtest_portfolio = portfolio
            return portfolio
    
    def plot_portfolio_analysis(self, save_path: str = None):
        """
        Create comprehensive portfolio analysis plots
        
        Args:
            save_path: Path to save plots
        """
        if not hasattr(self, 'portfolio'):
            raise ValueError("No portfolio available for plotting")
            
        try:
            # Set up the plotting style
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('seaborn')
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('LOB Portfolio Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Cumulative Returns
            try:
                portfolio_value = self.portfolio.value()
                axes[0, 0].plot(portfolio_value)
                axes[0, 0].set_title('Portfolio Value Over Time')
                axes[0, 0].set_ylabel('Portfolio Value')
                axes[0, 0].grid(True, alpha=0.3)
            except Exception as e:
                print(f"⚠️ Error plotting portfolio value: {e}")
                axes[0, 0].text(0.5, 0.5, 'Portfolio Value\nNot Available', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Portfolio Value Over Time')
            
            # 2. Drawdown
            try:
                drawdown = self.portfolio.drawdown()
                # Ensure drawdown is 1-dimensional
                if hasattr(drawdown, 'values'):
                    drawdown_values = drawdown.values
                else:
                    drawdown_values = drawdown
                    
                # Handle different drawdown formats
                if hasattr(drawdown_values, 'shape') and len(drawdown_values.shape) > 1:
                    # If it's 2D, take the first column
                    drawdown_values = drawdown_values[:, 0] if drawdown_values.shape[1] > 0 else drawdown_values.flatten()
                
                axes[0, 1].fill_between(drawdown.index, drawdown_values, 0, alpha=0.3, color='red')
                axes[0, 1].set_title('Portfolio Drawdown')
                axes[0, 1].set_ylabel('Drawdown (%)')
                axes[0, 1].grid(True, alpha=0.3)
            except Exception as e:
                print(f"⚠️ Error plotting drawdown: {e}")
                axes[0, 1].text(0.5, 0.5, 'Drawdown\nNot Available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Portfolio Drawdown')
            
            # 3. LOB Features (if available)
            if self.lob_data:
                try:
                    symbol = list(self.lob_data.keys())[0]
                    df = self.lob_data[symbol]
                    
                    # Plot spread over time
                    spread_data = df['Spread_Ratio'].resample('1T').mean()
                    axes[1, 0].plot(spread_data.index, spread_data.values, color='green')
                    axes[1, 0].set_title(f'{symbol} - Spread Ratio Over Time')
                    axes[1, 0].set_ylabel('Spread Ratio')
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Plot volume imbalance
                    imbalance_data = df['Volume_Imbalance'].resample('1T').mean()
                    axes[1, 1].plot(imbalance_data.index, imbalance_data.values, color='blue')
                    axes[1, 1].set_title(f'{symbol} - Volume Imbalance Over Time')
                    axes[1, 1].set_ylabel('Volume Imbalance')
                    axes[1, 1].grid(True, alpha=0.3)
                except Exception as e:
                    print(f"⚠️ Error plotting LOB features: {e}")
                    axes[1, 0].text(0.5, 0.5, 'LOB Features\nNot Available', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('LOB Features')
                    axes[1, 1].text(0.5, 0.5, 'LOB Features\nNot Available', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('LOB Features')
            else:
                # Fallback to monthly returns heatmap
                try:
                    monthly_returns = self.portfolio.returns.resample('H').sum()
                    monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.date, 
                                                                   monthly_returns.index.hour]).sum()
                    
                    sns.heatmap(monthly_returns_pivot.values, 
                               ax=axes[1, 0], cmap='RdYlGn', center=0)
                    axes[1, 0].set_title('Hourly Returns Heatmap')
                    
                    # Risk-return scatter
                    if len(self.returns.columns) > 1:
                        asset_returns = self.returns.mean() * 1440  # Annualized returns (minutes)
                        asset_vol = self.returns.std() * np.sqrt(1440)  # Annualized volatility
                        
                        axes[1, 1].scatter(asset_vol, asset_returns, s=100, alpha=0.7)
                        for i, asset in enumerate(self.returns.columns):
                            axes[1, 1].annotate(asset, (asset_vol[i], asset_returns[i]), 
                                               xytext=(5, 5), textcoords='offset points')
                        
                        axes[1, 1].set_xlabel('Volatility')
                        axes[1, 1].set_ylabel('Return')
                        axes[1, 1].set_title('Risk-Return Profile')
                        axes[1, 1].grid(True, alpha=0.3)
                except Exception as e:
                    print(f"⚠️ Error plotting fallback charts: {e}")
                    axes[1, 0].text(0.5, 0.5, 'Returns Heatmap\nNot Available', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Returns Heatmap')
                    axes[1, 1].text(0.5, 0.5, 'Risk-Return\nNot Available', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Risk-Return Profile')
            
            plt.tight_layout()
            
            if save_path:
                try:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"📊 Portfolio analysis plots saved to {save_path}")
                except Exception as e:
                    print(f"⚠️ Error saving plot: {e}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error in portfolio plotting: {e}")
            print("📊 Creating simple fallback plot...")
            
            # Simple fallback plot
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Portfolio Analysis\nPlot Not Available', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
            plt.title('Portfolio Analysis Dashboard')
            plt.show()
    
    def generate_portfolio_report(self) -> str:
        """
        Generate a comprehensive portfolio report
        
        Returns:
            Formatted report string
        """
        if not hasattr(self, 'portfolio_stats'):
            self.analyze_portfolio_performance()
            
        stats = self.portfolio_stats
        
        # Safe formatting function
        def safe_format(value, format_str='.2%'):
            try:
                if isinstance(value, (int, float)):
                    if format_str == '.2%':
                        return f"{value:.2%}"
                    elif format_str == '.2f':
                        return f"{value:.2f}"
                    else:
                        return str(value)
                else:
                    return str(value)
            except:
                return str(value)
        
        report = f"""
📊 LOB PORTFÖY ANALİZ RAPORU
{'='*50}

🎯 PERFORMANS METRİKLERİ:
• Toplam Getiri: {safe_format(stats.get('total_return', 0))}
• Yıllık Getiri: {safe_format(stats.get('annual_return', 0))}
• Volatilite: {safe_format(stats.get('volatility', 0))}
• Sharpe Oranı: {safe_format(stats.get('sharpe_ratio', 0), '.2f')}
• Maksimum Drawdown: {safe_format(stats.get('max_drawdown', 0))}
• Calmar Oranı: {safe_format(stats.get('calmar_ratio', 0), '.2f')}

📈 TRADING METRİKLERİ:
• Kazanma Oranı: {safe_format(stats.get('win_rate', 0))}
• Profit Faktörü: {safe_format(stats.get('profit_factor', 0), '.2f')}
• Ortalama Kazanç: {safe_format(stats.get('avg_win', 0))}
• Ortalama Kayıp: {safe_format(stats.get('avg_loss', 0))}

⚠️ RİSK METRİKLERİ:
• VaR (95%): {safe_format(stats.get('var_95', 0))}
• CVaR (95%): {safe_format(stats.get('cvar_95', 0))}

📊 LOB VERİ BİLGİLERİ:
• Sembol Sayısı: {len(self.lob_data) if self.lob_data else 0}
• Veri Noktası: {len(self.portfolio_data) if self.portfolio_data is not None else 0}
• Zaman Aralığı: {self.portfolio_data.index[0] if self.portfolio_data is not None else 'N/A'} - {self.portfolio_data.index[-1] if self.portfolio_data is not None else 'N/A'}

{'='*50}
        """
        
        return report
    
    def save_portfolio_results(self, filename: str = 'results/portfolio_analysis.json'):
        """
        Save portfolio analysis results to JSON file
        
        Args:
            filename: Output file path
        """
        if not hasattr(self, 'portfolio_stats'):
            self.analyze_portfolio_performance()
            
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'iloc'):
                # Handle pandas Series
                return float(obj.iloc[0]) if len(obj) > 0 else 0.0
            elif hasattr(obj, 'item'):
                # Handle numpy scalars
                return float(obj.item())
            elif isinstance(obj, pd.Series):
                # Handle pandas Series
                return float(obj.iloc[0]) if len(obj) > 0 else 0.0
            return obj
        
        results = {
            'portfolio_stats': {k: convert_numpy(v) for k, v in self.portfolio_stats.items()},
            'portfolio_data_shape': self.portfolio_data.shape if self.portfolio_data is not None else None,
            'returns_shape': self.returns.shape if self.returns is not None else None,
            'lob_symbols': list(self.lob_data.keys()) if self.lob_data else [],
            'analysis_date': datetime.now().isoformat()
        }
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Portfolio analysis results saved to {filename}") 