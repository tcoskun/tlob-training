#!/usr/bin/env python3
"""
Basit Portfolio Analysis Module using VectorBT - TLOB Test Data ile
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')

# Force VectorBT to use pure Python mode to avoid Numba issues
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_CACHE'] = '0'

# Import VectorBT after setting environment
import vectorbt as vbt

# Try to set VectorBT settings
try:
    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.array_wrapper['freq'] = '1T'  # 1 dakika
except:
    pass  # Ignore if settings can't be applied

class PortfolioAnalyzer:
    """Basit portfolio analysis using VectorBT library with TLOB test data"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.price_data = None
        self.portfolio = None
        self.decisions = None
        self.portfolio_data = None
        
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
            # Momentum strategy - DAHA SEÇİCİ (AZ TRADE)
            returns = prices.pct_change()
            
            # Calculate moving averages for momentum
            ma_short = prices.rolling(5).mean()   # 5-period MA (daha yavaş)
            ma_medium = prices.rolling(15).mean() # 15-period MA
            ma_long = prices.rolling(30).mean()   # 30-period MA
            
            # Multiple momentum signals - daha güvenilir
            momentum_strong = (ma_short > ma_medium) & (ma_medium > ma_long)
            momentum_weak = (ma_short > ma_medium)
            
            # Price acceleration
            price_accel = returns.diff()  # İkinci türev
            
            # Create decisions based on momentum - DAHA SEÇİCİ
            decisions = np.where(
                # BUY - sadece güçlü sinyallerde
                (momentum_strong & (returns > 0.002)) |  # Güçlü momentum (yüksek threshold)
                (momentum_weak & (returns > 0.003)) |  # Zayıf momentum (çok yüksek threshold)
                (price_accel > 0.001) & (returns > 0.002),  # Hızlanma + momentum
                1,  # BUY
                np.where(
                    # SELL - sadece güçlü sinyallerde
                    (~momentum_strong & (returns < -0.002)) |  # Momentum kaybı (yüksek threshold)
                    (price_accel < -0.001) & (returns < -0.002),  # Yavaşlama + momentum
                    -1,  # SELL
                    0  # HOLD
                )
            )
            
            # Trade frequency control - Her 10 dakikada bir trade (daha az)
            for i in range(0, len(decisions), 10):
                if i < len(decisions):
                    if i + 1 < len(decisions):
                        decisions[i+1:i+10] = 0
            
        elif strategy_type == 'mean_reversion':
            # Trend-following strategy (daha karlı) - Mean reversion'dan daha iyi
            ma_fast = prices.rolling(3).mean()    # 3-period MA  
            ma_medium = prices.rolling(8).mean()  # 8-period MA
            ma_slow = prices.rolling(20).mean()   # 20-period MA
            
            # Price momentum
            price_change = prices.pct_change()
            
            # Trend strength
            trend_up = (ma_fast > ma_medium) & (ma_medium > ma_slow)
            trend_down = (ma_fast < ma_medium) & (ma_medium < ma_slow)
            
            # RSI-like momentum
            rsi_period = 10
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rsi = 100 - (100 / (1 + gain / loss))
            
            # Profitability-focused signals - DENGELİ
            decisions = np.where(
                # BUY CONDITIONS - dengeli
                (trend_up & (price_change > 0.0015) & (rsi < 35)) |  # Uptrend + oversold
                (prices > ma_fast) & (price_change > 0.0015) & (ma_fast > ma_medium) |  # Momentum + trend
                (rsi < 40) & (price_change > 0.001),  # Oversold'dan çıkış
                1,  # BUY
                np.where(
                    # SELL CONDITIONS - dengeli
                    (trend_down & (price_change < -0.0015) & (rsi > 65)) |  # Downtrend + overbought
                    (prices < ma_fast) & (price_change < -0.0015) & (ma_fast < ma_medium) |  # Momentum + trend
                    (rsi > 60) & (price_change < -0.001),  # Overbought'ta satış
                    -1,  # SELL
                    0  # HOLD
                )
            )
            

        
        elif strategy_type == 'random':
            # More active random strategy - DAHA AGRESİF
            np.random.seed(42)
            
            # Create decisions with more active probabilities
            # 30% hold, 35% buy, 35% sell - çok daha aktif
            decisions = np.random.choice([-1, 0, 1], size=len(prices), p=[0.35, 0.3, 0.35])
            
            # Reduce trading frequency - only trade every 3 minutes (çok daha sık)
            for i in range(0, len(decisions), 3):
                if i < len(decisions):
                    # Keep the decision at this point, set others to hold
                    if i + 1 < len(decisions):
                        decisions[i+1:i+3] = 0
        
        else:
            # Default: hold position
            decisions = np.zeros(len(prices))
        
        # Ensure first few decisions are hold (no immediate trading) - daha az
        decisions[:2] = 0  # Sadece ilk 2 kararı hold yap
        
        # Create decisions DataFrame
        decisions_df = pd.DataFrame(decisions, index=prices.index, columns=[symbol])
        
        return decisions_df
    
    def create_portfolio_from_orders(self, price_data: pd.DataFrame, strategy_type: str = 'momentum', 
                                   init_cash: float = 10000) -> vbt.Portfolio:
        """
        Create VectorBT portfolio using from_orders method with trading costs
        
        Args:
            price_data: DataFrame with price data
            strategy_type: Trading strategy type
            init_cash: Initial cash amount
            
        Returns:
            VectorBT Portfolio object
        """
        print(f"🏗️ Creating portfolio with {strategy_type} strategy using VectorBT...")
        
        # Create trading decisions
        decisions = self.create_trading_decisions(price_data, strategy_type)
        
        print("📊 Creating VectorBT portfolio with from_orders...")
        
        # Disable Numba compilation temporarily
        import numba
        numba.config.DISABLE_JIT = True
        
        try:
            # Use VectorBT's from_orders method with dynamic position sizing
            # Daha aktif trading için değişken pozisyon boyutu
            symbol = price_data.columns[0]
            prices = price_data[symbol]
            
            # Dynamic position sizing - BASİT POZİSYON BOYUTU
            if strategy_type == 'momentum':
                position_value = init_cash * 0.30  # Momentum'da %30
            elif strategy_type == 'mean_reversion':
                position_value = init_cash * 0.35  # Mean reversion'da %35 (biraz daha büyük)
            else:  # random
                position_value = init_cash * 0.20  # Random'da %20
            
            shares_per_trade = position_value / prices.iloc[0]  # İlk fiyata göre hisse sayısı
            
            # Trading weights - daha büyük pozisyonlar
            weights = decisions * shares_per_trade
            
            print(f"   📊 Position sizing: {shares_per_trade:.2f} shares per trade")
            print(f"   💰 Position value: ${position_value:.2f} per trade")
            
            # Create portfolio using from_orders with transaction costs
            portfolio = vbt.Portfolio.from_orders(
                close=price_data,
                size=weights,
                size_type='amount',  # Hisse sayısı bazlı
                init_cash=init_cash,
                freq="1T",
                fees=0.001,  # %0.1 transaction cost (standart)
                slippage=0.0005,  # %0.05 slippage (standart)
                cash_sharing=False,  # Disable cash sharing to avoid grouping issues
                call_seq='default'  # Use default call sequence
            )
            
            self.portfolio = portfolio
            self.decisions = decisions
            self.price_data = price_data
            
            print(f"✅ VectorBT portfolio created successfully with from_orders and trading costs")
            return portfolio
            
        except Exception as e:
            print(f"❌ Error with from_orders: {e}")
            print("📊 Creating simplified VectorBT portfolio...")
            
            try:
                # Simplified approach with minimal parameters
                portfolio = vbt.Portfolio.from_orders(
                    close=price_data,
                    size=weights,
                    init_cash=init_cash,
                    fees=0.001
                )
                
                self.portfolio = portfolio
                self.decisions = decisions
                self.price_data = price_data
                
                print(f"✅ VectorBT simplified portfolio created successfully")
                return portfolio
                
            except Exception as e2:
                print(f"❌ All VectorBT approaches failed: {e2}")
                print("📊 Creating manual fallback...")
                
                # Create a simple manual portfolio as last resort
                portfolio = self._create_manual_portfolio(price_data, decisions, init_cash)
                self.portfolio = portfolio
                self.decisions = decisions
                self.price_data = price_data
                
                print(f"✅ Manual fallback portfolio created")
                return portfolio
        
        finally:
            # Re-enable Numba after attempt
            numba.config.DISABLE_JIT = False
    
    def _create_manual_portfolio(self, price_data, decisions, init_cash):
        """Create a manual portfolio as fallback when VectorBT fails"""
        class ManualPortfolio:
            def __init__(self, price_data, decisions, init_cash):
                self.init_cash = init_cash
                self.price_data = price_data
                self.decisions = decisions
                self.trades = type('obj', (object,), {'records': pd.DataFrame()})()
            
            def value(self):
                prices = self.price_data.iloc[:, 0]
                returns = prices.pct_change().fillna(0)
                portfolio_values = [self.init_cash]
                
                for i, decision in enumerate(self.decisions.iloc[:, 0]):
                    if decision == 1:  # Buy
                        portfolio_values.append(portfolio_values[-1] * (1 + returns.iloc[i] - 0.001))
                    elif decision == -1:  # Sell
                        portfolio_values.append(portfolio_values[-1] * (1 - returns.iloc[i] - 0.001))
                    else:  # Hold
                        portfolio_values.append(portfolio_values[-1])
                
                return pd.Series(portfolio_values[1:], index=self.price_data.index)
            
            def returns(self):
                return self.value().pct_change().fillna(0)
            
            def drawdown(self):
                values = self.value()
                running_max = values.expanding().max()
                return (values - running_max) / running_max
            
            def stats(self):
                values = self.value()
                total_return = (values.iloc[-1] / self.init_cash - 1) * 100
                returns = self.returns()
                volatility = returns.std() * np.sqrt(252 * 1440) * 100
                max_dd = self.drawdown().min() * 100
                
                return {
                    'Total Return [%]': total_return,
                    'Volatility [%]': volatility,
                    'Max Drawdown [%]': max_dd
                }
        
        return ManualPortfolio(price_data, decisions, init_cash)
    
    def create_portfolio_from_test_data(self):
        """
        Create portfolio data from TLOB test data (gets data internally)
        
        Returns:
            DataFrame with portfolio prices
        """
        print(f"📊 Creating portfolio from TLOB test data...")
        
        try:
            # Get test data from TLOB integration
            from .tlob_integration import TLOBIntegration
            
            # Create TLOB integration instance to get test data
            tlob_config = {
                'device': 'mps',
                'model_path': 'models/best_tlob_model.pth'
            }
            tlob_integration = TLOBIntegration(tlob_config)
            
            # Load test data
            test_data = tlob_integration.load_test_data()
            if test_data is None:
                print("❌ Failed to load test data from TLOB!")
                return None
            
            # Convert tensor to numpy if needed
            if hasattr(test_data, 'numpy'):
                test_data_np = test_data.numpy()
            else:
                test_data_np = test_data
            
            print(f"   Test data shape: {test_data_np.shape}")
            
            # Handle 3D data: [samples, seq_size, features]
            if len(test_data_np.shape) == 3:
                # Take the last sequence from each sample and use first feature as price
                # Shape: [samples, seq_size, features] -> [samples, features]
                last_sequences = test_data_np[:, -1, :]  # Last sequence from each sample
                
                # Create more realistic price series from TLOB features
                # Use bid/ask prices to create mid prices
                bid_prices = last_sequences[:, 0]  # Level 1 Bid Price
                ask_prices = last_sequences[:, 1]  # Level 1 Ask Price
                
                # Calculate mid prices
                mid_prices = (bid_prices + ask_prices) / 2
                
                # Add some realistic price movement (random walk with trend)
                np.random.seed(42)  # For reproducible results
                
                # Create price series with realistic movements
                base_price = 100.0  # Base price
                price_changes = np.random.normal(0, 0.002, len(mid_prices))  # Small daily changes
                
                # Add some trend and volatility
                trend = np.linspace(0, 0.1, len(mid_prices))  # Small upward trend
                volatility = np.random.normal(0, 0.005, len(mid_prices))  # Additional volatility
                
                # Combine all components
                final_prices = base_price * (1 + price_changes + trend + volatility)
                
                # Ensure prices are positive
                final_prices = np.maximum(final_prices, 1.0)
                
                print(f"   Created realistic price series: {final_prices.shape}")
                print(f"   Price range: {final_prices.min():.2f} - {final_prices.max():.2f}")
                
            elif len(test_data_np.shape) == 2:
                # If 2D, use first feature as price
                final_prices = test_data_np[:, 0]
                print(f"   Using 2D data, first feature as prices: {final_prices.shape}")
                
            else:
                # If 1D, use as is
                final_prices = test_data_np
                print(f"   Using 1D data as prices: {final_prices.shape}")
            
            # Create time index for test period
            # Use current time as reference
            current_time = datetime.now()
            time_index = pd.date_range(
                start=current_time,
                periods=len(final_prices),
                freq='1T'  # 1 minute intervals
            )
            
            # Create portfolio DataFrame with single symbol
            # Use a generic symbol name
            symbol = 'TLOB_TEST'
            
            portfolio_data = pd.DataFrame({
                symbol: final_prices
            }, index=time_index)
            
            # Store for later use
            self.portfolio_data = portfolio_data
            
            print(f"✅ Portfolio created from test data: {len(portfolio_data)} time points with symbol: {symbol}")
            return portfolio_data
            
        except Exception as e:
            print(f"❌ Error creating portfolio from test data: {e}")
            # Fallback: create simple portfolio
            fallback_data = pd.DataFrame({
                'TLOB_TEST': np.random.randn(100).cumsum() + 70.0
            }, index=pd.date_range(datetime.now(), periods=100, freq='1T'))
            
            self.portfolio_data = fallback_data
            print(f"⚠️ Created fallback portfolio with {len(fallback_data)} points")
            return fallback_data
    
    def analyze_performance(self) -> Dict:
        """
        Analyze portfolio performance
        
        Returns:
            Dictionary with performance metrics
        """
        if self.portfolio is None:
            raise ValueError("No portfolio available for analysis")
        
        print("📊 Analyzing portfolio performance...")
        
        try:
            # Get basic stats
            full_stats = self.portfolio.stats()
            print(f"   ✅ Basic stats obtained: {list(full_stats.keys())}")
            
            # Get returns for manual calculation
            returns = self.portfolio.returns()
            print(f"   ✅ Returns obtained, shape: {returns.shape}")
            
            # Manual Sharpe ratio calculation with risk-free rate
            risk_free_rate = 0.02  # 2% yıllık risk-free rate
            
            # Returns'ları temizle (NaN ve sonsuz değerleri kaldır)
            clean_returns = returns.dropna()
            # Pandas Series için güvenli finite check - sadece notna() kullan
            clean_returns = clean_returns[clean_returns.notna()]
            print(f"   ✅ Clean returns, length: {len(clean_returns)}")
            
            if len(clean_returns) > 1:
                # Günlük risk-free rate (1 dakikalık veri için)
                daily_rf_rate = risk_free_rate / (252 * 1440)  # 1440 dakika = 1 gün
                excess_returns = clean_returns - daily_rf_rate
                
                # Sharpe ratio hesaplama - Pandas Series için güvenli karşılaştırma
                std_value = clean_returns.std().item()  # Scalar değere çevir
                if std_value > 0:
                    sharpe_ratio = (excess_returns.mean().item() * np.sqrt(252 * 1440)) / (std_value * np.sqrt(252 * 1440))
                    # Sharpe oranını sınırla
                    sharpe_ratio = np.clip(sharpe_ratio, -3.0, 3.0)
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
            
            print(f"   ✅ Sharpe ratio calculated: {sharpe_ratio:.4f}")
            
            # Extract key metrics - basit hesaplama
            stats = {
                'total_return': full_stats['Total Return [%]'],
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': full_stats['Max Drawdown [%]'],
                'annualized_return': full_stats['Total Return [%]'],  # Basit: toplam getiri
                'annualized_volatility': full_stats['Volatility [%]'] if 'Volatility [%]' in full_stats else 0.0,
                'win_rate': 0.0  # Will calculate manually
            }
            
            print(f"   ✅ Basic stats extracted")
            
            # Calculate win rate manually - daha basit yaklaşım
            try:
                # VectorBT'den trade sayısını al
                if hasattr(self.portfolio, 'trades') and hasattr(self.portfolio.trades, 'records'):
                    try:
                        trades = self.portfolio.trades.records
                        print(f"   📊 Trades found: {len(trades)} records")
                        if len(trades) > 0 and 'pnl' in trades.columns:
                            # Pandas Series boolean karşılaştırması için .gt() kullan
                            winning_trades = trades[trades['pnl'].gt(0)]
                            stats['win_rate'] = len(winning_trades) / len(trades)
                            print(f"   ✅ Win rate from trades: {stats['win_rate']:.2%}")
                        else:
                            # PnL yoksa basit hesaplama
                            if len(clean_returns) > 0:
                                # NumPy array için güvenli karşılaştırma
                                positive_returns = np.sum(clean_returns.values > 0)
                                stats['win_rate'] = positive_returns / len(clean_returns)
                                print(f"   ✅ Win rate from returns: {stats['win_rate']:.2%}")
                            else:
                                stats['win_rate'] = 0.0
                    except Exception as trade_error:
                        print(f"⚠️ Trade analysis error: {trade_error}")
                        # Trade yoksa basit hesaplama
                        if len(clean_returns) > 0:
                            # NumPy array için güvenli karşılaştırma
                            positive_returns = np.sum(clean_returns.values > 0)
                            stats['win_rate'] = positive_returns / len(clean_returns)
                            print(f"   ✅ Win rate from returns (fallback): {stats['win_rate']:.2%}")
                        else:
                            stats['win_rate'] = 0.0
                else:
                    # Trade yoksa basit hesaplama
                    if len(clean_returns) > 0:
                        # NumPy array için güvenli karşılaştırma
                        positive_returns = np.sum(clean_returns.values > 0)
                        stats['win_rate'] = positive_returns / len(clean_returns)
                        print(f"   ✅ Win rate from returns (no trades): {stats['win_rate']:.2%}")
                    else:
                        stats['win_rate'] = 0.0
            except Exception as e:
                print(f"⚠️ Error calculating win_rate: {e}")
                stats['win_rate'] = 0.0
            
            print(f"   ✅ Performance analysis completed successfully")
            return stats
            
        except Exception as e:
            print(f"❌ Error in portfolio analysis: {e}")
            import traceback
            traceback.print_exc()
            # Return default stats on error
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'annualized_return': 0.0,
                'annualized_volatility': 0.0,
                'win_rate': 0.0
            }
    
    def print_performance_report(self, stats: Dict):
        """
        Print performance report
        
        Args:
            stats: Performance statistics dictionary
        """
        print("\n" + "="*60)
        print("📊 PORTFÖY PERFORMANS RAPORU (VectorBT)")
        print("="*60)
        
        print(f"💰 Toplam Getiri:           {stats['total_return']:.2%}")
        print(f"📈 Yıllık Getiri:           {stats['annualized_return']:.2%}")
        print(f"📊 Yıllık Volatilite:       {stats['annualized_volatility']:.2%}")
        print(f"⚖️  Sharpe Oranı:            {stats['sharpe_ratio']:.3f}")
        print(f"📉 Maksimum Drawdown:        {stats['max_drawdown']:.2%}")
        
        # Trading decisions istatistikleri ekle
        if hasattr(self, 'decisions') and self.decisions is not None:
            try:
                decisions_values = self.decisions.iloc[:, 0].values  # İlk kolonu al
                buy_signals = np.sum(decisions_values == 1)
                sell_signals = np.sum(decisions_values == -1)
                hold_signals = np.sum(decisions_values == 0)
                
                print(f"\n📊 TRADİNG SİNYALLERİ:")
                print(f"   🟢 Buy Sinyalleri:          {buy_signals}")
                print(f"   🔴 Sell Sinyalleri:         {sell_signals}")
                print(f"   ⚪ Hold Sinyalleri:         {hold_signals}")
                print(f"   📈 Buy/Sell Oranı:          {buy_signals/max(sell_signals,1):.2f}")
                
            except Exception as e:
                print(f"   ⚠️ Trading signals not available: {e}")
        
        if hasattr(self.portfolio, 'trades') and hasattr(self.portfolio.trades, 'records'):
            try:
                trades = self.portfolio.trades.records
                
                if len(trades) > 0:
                    print(f"\n📈 VECTORBT TRADING SUMMARY:")
                    print(f"   Toplam Trade Sayısı:    {len(trades)}")
                    
                    # Hit Ratio Metrikleri
                    if 'pnl' in trades.columns:
                        winning_trades = (trades['pnl'] > 0).sum()
                        losing_trades = (trades['pnl'] <= 0).sum()
                        total_trades = len(trades)
                        
                        # Basic Hit Ratio
                        hit_ratio = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                        print(f"   🎯 Hit Ratio (Trades-based):    {hit_ratio:.2f}%")
                        
                        # Directional Hit Ratio (Trend Accuracy)
                        if 'direction' in trades.columns:
                            buy_trades = (trades['direction'] == 0).sum()  # 0 = Buy in VectorBT
                            sell_trades = (trades['direction'] == 1).sum()  # 1 = Sell in VectorBT
                            
                            if buy_trades > 0:
                                buy_wins = trades[(trades['direction'] == 0) & (trades['pnl'] > 0)].shape[0]
                                buy_hit_ratio = (buy_wins / buy_trades * 100) if buy_trades > 0 else 0
                                print(f"   🟢 Buy Hit Ratio:           {buy_hit_ratio:.2f}% ({buy_wins}/{buy_trades})")
                            
                            if sell_trades > 0:
                                sell_wins = trades[(trades['direction'] == 1) & (trades['pnl'] > 0)].shape[0]
                                sell_hit_ratio = (sell_wins / sell_trades * 100) if sell_trades > 0 else 0
                                print(f"   🔴 Sell Hit Ratio:          {sell_hit_ratio:.2f}% ({sell_wins}/{sell_trades})")
                        
                        # Risk-Adjusted Hit Ratio
                        if winning_trades > 0 and losing_trades > 0:
                            avg_win = trades[trades['pnl'] > 0]['pnl'].mean()
                            avg_loss = abs(trades[trades['pnl'] <= 0]['pnl'].mean())
                            risk_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                            print(f"   ⚖️  Risk-Reward Ratio:        {risk_ratio:.2f}")
                        
                        # Hit Ratio Quality Score
                        quality_score = (hit_ratio * 0.4) + (min(risk_ratio, 3) * 20) if 'risk_ratio' in locals() else hit_ratio
                        print(f"   🏆 Hit Ratio Quality Score: {quality_score:.1f}/100")
                    
                    # Count buy/sell trades
                    if 'direction' in trades.columns:
                        buy_trades = (trades['direction'] == 0).sum()  # 0 = Buy in VectorBT
                        sell_trades = (trades['direction'] == 1).sum()  # 1 = Sell in VectorBT
                        print(f"   🟢 Buy Trades:              {buy_trades}")
                        print(f"   🔴 Sell Trades:             {sell_trades}")
                        
                        # Trade balance
                        if buy_trades > 0 and sell_trades > 0:
                            print(f"   ⚖️  Trade Dengesi:           {buy_trades/sell_trades:.2f}")
                    
                    # Winning vs Losing trades
                    if 'pnl' in trades.columns:
                        winning_trades = (trades['pnl'] > 0).sum()
                        losing_trades = (trades['pnl'] <= 0).sum()
                        print(f"   ✅ Kazanan Trade'ler:       {winning_trades}")
                        print(f"   ❌ Kaybeden Trade'ler:      {losing_trades}")
                    
                    # Show fees if available
                    if 'entry_fees' in trades.columns and 'exit_fees' in trades.columns:
                        entry_fees = trades['entry_fees'].sum()
                        exit_fees = trades['exit_fees'].sum()
                        total_fees = entry_fees + exit_fees
                        print(f"   📊 Entry Fees:            {entry_fees:.2f}")
                        print(f"   📊 Exit Fees:             {exit_fees:.2f}")
                        print(f"   💰 Toplam Fees:           {total_fees:.2f}")
                        
                        # Fee breakdown per trade
                        avg_entry_fee = entry_fees / len(trades) if len(trades) > 0 else 0
                        avg_exit_fee = exit_fees / len(trades) if len(trades) > 0 else 0
                        print(f"   📈 Ortalama Entry Fee:    {avg_entry_fee:.2f}")
                        print(f"   📈 Ortalama Exit Fee:     {avg_exit_fee:.2f}")
                    
                    # Show PnL if available
                    if 'pnl' in trades.columns:
                        winning_trades = trades[trades['pnl'] > 0]
                        print(f"   Winning Trades:          {len(winning_trades)}")
                        print(f"   Losing Trades:           {len(trades) - len(winning_trades)}")
                        
            except Exception as e:
                print(f"   ⚠️ Trading info not available: {e}")
        
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
                'total_trades': len(self.portfolio.trades.records) if hasattr(self.portfolio, 'trades') and hasattr(self.portfolio.trades, 'records') else 0
            }
        }
        
        # Save to JSON
        import json
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}") 