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
import pickle

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

class LSTMPortfolioAnalyzer:
    """Portfolio analysis using VectorBT library with LSTM test data"""

    def __init__(self):
        self.price_data = None
        self.decisions = None
        self.analysis_results = None,
        self.start_index = None,
        self.y_hat_last = None,
        self.SEQ_LEN = None

    def readModel(self):
        try:
          # Proje kök dizinini bul
          import os
          project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
          tensorflow_results_path = os.path.join(project_root, 'models', 'tensorflow_results.pkl')
          
          with open(tensorflow_results_path, 'rb') as f:
              tensorflow_results = pickle.load(f)
          print("Successfully loaded TensorFlow results")
        except FileNotFoundError:
                print(f"Error: '{tensorflow_results_path}' not found. Please run tensorflow_part.py first.")
                return
        except Exception as e:
                print(f"Error loading TensorFlow results: {e}")
                return
            
        self.price_data = tensorflow_results['df']
        self.y_hat_last = tensorflow_results['y_hat_last']
        self.SEQ_LEN = tensorflow_results['SEQ_LEN']

        print(f"DataFrame shape: {self.price_data.shape}")
        print(f"Predicted prices length: {len(self.y_hat_last)}")
        print(f"Sequence length: {self.SEQ_LEN}")
        
        self.start_index = self.SEQ_LEN + int(0.5 * (len(self.price_data) - self.SEQ_LEN))

    def create_trading_decisions(self, strategy_type):
        self.strategy_type = strategy_type

        MOMENTUM_CONFIG = {
            'min_threshold': 0.0005,     
            'trend_window': 3,           
            'volatility_max': 0.01, 
            'volatility_window': 10,
            'min_trade_interval': 5,
            'profit_take_threshold': 0.002,
            'stop_loss_threshold': 0.001
        }
        
        MEAN_REVERSION_CONFIG = {
            'min_threshold': 0.0005,    
            'ma_window': 20,      
            'confirmation_window': 3, 
            'min_trade_interval': 8,
            'profit_take_threshold': 0.003,
            'stop_loss_threshold': 0.002,
            'max_holding_periods': 50    
        }
        
        VOLATILITY_BREAKOUT_CONFIG = {
            'volatility_window': 20, 
            'breakout_factor': 8,   
            'min_trade_interval': 10,   
            'profit_take_threshold': 0.004, 
            'stop_loss_threshold': 0.002, 
            'confirmation_required': True, 
            'confirmation_window': 2
        }
        
        decisions = pd.Series(0, index=self.price_data.index, dtype=int)
    
        test_end_index = self.start_index + len(self.y_hat_last)
        if test_end_index > len(self.price_data):
            test_end_index = len(self.price_data)
            self.y_hat_last = self.y_hat_last[:len(self.price_data) - self.start_index]
    
        test_indices = self.price_data.index[self.start_index:test_end_index]
    
        if strategy_type == 'momentum':
            prev_prices = self.price_data['midPrice'].shift(1).loc[test_indices]
            
            price_changes = self.y_hat_last - prev_prices.values
            price_change_pct = np.abs(price_changes) / prev_prices.values
            
            threshold_filter = price_change_pct > MOMENTUM_CONFIG['min_threshold']
            
            trend_window = MOMENTUM_CONFIG['trend_window']
            buy_trend = np.zeros_like(price_changes, dtype=bool)
            sell_trend = np.zeros_like(price_changes, dtype=bool)
            
            for i in range(trend_window, len(price_changes)):
                if all(price_changes[i-j] > 0 for j in range(1, trend_window + 1)):
                    buy_trend[i] = True
                elif all(price_changes[i-j] < 0 for j in range(1, trend_window + 1)):
                    sell_trend[i] = True
            
            volatility_window = MOMENTUM_CONFIG['volatility_window']
            rolling_std = self.price_data['midPrice'].rolling(window=volatility_window).std()
            volatility_values = rolling_std.loc[test_indices]
            price_values = self.price_data['midPrice'].loc[test_indices]
            volatility_filter = (volatility_values.values / price_values.values) < MOMENTUM_CONFIG['volatility_max']
            
            min_trade_interval = MOMENTUM_CONFIG['min_trade_interval']
            last_trade_index = -min_trade_interval - 1
            
            # Combine all filters
            buy_signals = (
                (price_changes > 0) & 
                threshold_filter & 
                buy_trend & 
                volatility_filter
            )
            
            sell_signals = (
                (price_changes < 0) & 
                threshold_filter & 
                sell_trend & 
                volatility_filter
            )
            
            for i in range(len(buy_signals)):
                if buy_signals[i] or sell_signals[i]:
                    if i - last_trade_index < min_trade_interval:
                        buy_signals[i] = False
                        sell_signals[i] = False
                    else:
                        last_trade_index = i
            
            current_position = 0  # 0: no position, 1: long, -1: short
            entry_price = 0
            entry_index = -1
            
            for i in range(len(buy_signals)):
                current_price = self.price_data['midPrice'].iloc[self.start_index + i]
                
                if current_position != 0:
                    if current_position == 1: 
                        profit_pct = (current_price - entry_price) / entry_price
                        if profit_pct >= MOMENTUM_CONFIG['profit_take_threshold'] or profit_pct <= -MOMENTUM_CONFIG['stop_loss_threshold']:
                            sell_signals[i] = -1
                            buy_signals[i] = False
                            current_position = 0
                            entry_price = 0
                            entry_index = -1
                            continue
                    
                    elif current_position == -1:
                        profit_pct = (entry_price - current_price) / entry_price
                        if profit_pct >= MOMENTUM_CONFIG['profit_take_threshold'] or profit_pct <= -MOMENTUM_CONFIG['stop_loss_threshold']:
                            buy_signals[i] = 1
                            sell_signals[i] = False
                            current_position = 0
                            entry_price = 0
                            entry_index = -1
                            continue
                
                if current_position == 0:
                    if buy_signals[i]:
                        current_position = 1
                        entry_price = current_price
                        entry_index = i
                    elif sell_signals[i]:
                        current_position = -1
                        entry_price = current_price
                        entry_index = i
            
            decisions.loc[test_indices] = np.select(
                [buy_signals, sell_signals],
                [1, -1],
                default=0
            )
        
        elif strategy_type == 'mean_reversion':
            ma_window = MEAN_REVERSION_CONFIG['ma_window']
            min_threshold = MEAN_REVERSION_CONFIG['min_threshold']
            confirmation_window = MEAN_REVERSION_CONFIG['confirmation_window']
            min_trade_interval = MEAN_REVERSION_CONFIG['min_trade_interval']
            max_holding_periods = MEAN_REVERSION_CONFIG['max_holding_periods']
            
            moving_avg = self.price_data['midPrice'].rolling(window=ma_window).mean()
            ma_values = moving_avg.loc[test_indices]
            predicted_prices = self.y_hat_last
            
            deviations = np.abs(predicted_prices - ma_values.values) / ma_values.values
            
            threshold_filter = deviations > min_threshold
            
            initial_buy_signals = (predicted_prices < ma_values.values) & threshold_filter
            initial_sell_signals = (predicted_prices > ma_values.values) & threshold_filter
            
            buy_signals = np.zeros_like(initial_buy_signals, dtype=bool)
            sell_signals = np.zeros_like(initial_sell_signals, dtype=bool)
            
            for i in range(confirmation_window, len(initial_buy_signals)):
                if all(initial_buy_signals[i-j] for j in range(confirmation_window)):
                    buy_signals[i] = True
                if all(initial_sell_signals[i-j] for j in range(confirmation_window)):
                    sell_signals[i] = True
            
            last_trade_index = -min_trade_interval - 1
            for i in range(len(buy_signals)):
                if buy_signals[i] or sell_signals[i]:
                    if i - last_trade_index < min_trade_interval:
                        buy_signals[i] = False
                        sell_signals[i] = False
                    else:
                        last_trade_index = i
            
            current_position = 0
            entry_price = 0
            entry_index = -1
            holding_periods = 0
            
            for i in range(len(buy_signals)):
                current_price = self.price_data['midPrice'].iloc[self.start_index + i]
                
                if current_position != 0:
                    holding_periods += 1
                    
                    if current_position == 1:
                        profit_pct = (current_price - entry_price) / entry_price
                        if (profit_pct >= MEAN_REVERSION_CONFIG['profit_take_threshold'] or 
                            profit_pct <= -MEAN_REVERSION_CONFIG['stop_loss_threshold'] or
                            holding_periods >= max_holding_periods):
                            sell_signals[i] = True
                            buy_signals[i] = False
                            current_position = 0
                            entry_price = 0
                            entry_index = -1
                            holding_periods = 0
                            continue
                    
                    elif current_position == -1:
                        profit_pct = (entry_price - current_price) / entry_price
                        if (profit_pct >= MEAN_REVERSION_CONFIG['profit_take_threshold'] or 
                            profit_pct <= -MEAN_REVERSION_CONFIG['stop_loss_threshold'] or
                            holding_periods >= max_holding_periods):
                            buy_signals[i] = True
                            sell_signals[i] = False
                            current_position = 0
                            entry_price = 0
                            entry_index = -1
                            holding_periods = 0
                            continue
                
                if current_position == 0:
                    if buy_signals[i]:
                        current_position = 1
                        entry_price = current_price
                        entry_index = i
                        holding_periods = 0
                    elif sell_signals[i]:
                        current_position = -1
                        entry_price = current_price
                        entry_index = i
                        holding_periods = 0
            
            decisions.loc[test_indices] = np.select(
                [buy_signals, sell_signals],
                [1, -1],
                default=0
            )
        
        elif strategy_type == 'volatility_breakout':
            volatility_window = VOLATILITY_BREAKOUT_CONFIG['volatility_window']
            breakout_factor = VOLATILITY_BREAKOUT_CONFIG['breakout_factor']
            min_trade_interval = VOLATILITY_BREAKOUT_CONFIG['min_trade_interval']
            confirmation_required = VOLATILITY_BREAKOUT_CONFIG['confirmation_required']
            confirmation_window = VOLATILITY_BREAKOUT_CONFIG['confirmation_window']
            
            moving_avg = self.price_data['midPrice'].rolling(window=volatility_window).mean()
            std_dev = self.price_data['midPrice'].rolling(window=volatility_window).std()
            
            upper_band = moving_avg + std_dev * breakout_factor
            lower_band = moving_avg - std_dev * breakout_factor
            
            upper_values = upper_band.loc[test_indices]
            lower_values = lower_band.loc[test_indices]
            predicted_prices = self.y_hat_last
            
            initial_buy_signals = predicted_prices > upper_values.values
            initial_sell_signals = predicted_prices < lower_values.values
            
            if confirmation_required:
                buy_signals = np.zeros_like(initial_buy_signals, dtype=bool)
                sell_signals = np.zeros_like(initial_sell_signals, dtype=bool)
                
                for i in range(confirmation_window, len(initial_buy_signals)):
                    if all(initial_buy_signals[i-j] for j in range(confirmation_window)):
                        buy_signals[i] = True
                    if all(initial_sell_signals[i-j] for j in range(confirmation_window)):
                        sell_signals[i] = True
            else:
                buy_signals = initial_buy_signals.copy()
                sell_signals = initial_sell_signals.copy()
            
            last_trade_index = -min_trade_interval - 1
            for i in range(len(buy_signals)):
                if buy_signals[i] or sell_signals[i]:
                    if i - last_trade_index < min_trade_interval:
                        buy_signals[i] = False
                        sell_signals[i] = False
                    else:
                        last_trade_index = i
            
            current_position = 0
            entry_price = 0
            entry_index = -1
            
            for i in range(len(buy_signals)):
                current_price = self.price_data['midPrice'].iloc[self.start_index + i]
                
                if current_position != 0:
                    if current_position == 1:
                        profit_pct = (current_price - entry_price) / entry_price
                        if (profit_pct >= VOLATILITY_BREAKOUT_CONFIG['profit_take_threshold'] or 
                            profit_pct <= -VOLATILITY_BREAKOUT_CONFIG['stop_loss_threshold']):
                            sell_signals[i] = True
                            buy_signals[i] = False
                            current_position = 0
                            entry_price = 0
                            entry_index = -1
                            continue
                    
                    elif current_position == -1:
                        profit_pct = (entry_price - current_price) / entry_price
                        if (profit_pct >= VOLATILITY_BREAKOUT_CONFIG['profit_take_threshold'] or 
                            profit_pct <= -VOLATILITY_BREAKOUT_CONFIG['stop_loss_threshold']):
                            buy_signals[i] = True
                            sell_signals[i] = False
                            current_position = 0
                            entry_price = 0
                            entry_index = -1
                            continue
                
                if current_position == 0:
                    if buy_signals[i]:
                        current_position = 1
                        entry_price = current_price
                        entry_index = i
                    elif sell_signals[i]:
                        current_position = -1
                        entry_price = current_price
                        entry_index = i
            
            decisions.loc[test_indices] = np.select(
                [buy_signals, sell_signals],
                [1, -1],
                default=0
            )
        
        self.decisions = decisions

    def analyze_decisions(self):
        print(f"\n=== Trading Decision Analysis for {self.strategy_type.upper()} Strategy ===")
        
        buy_count = (self.decisions == 1).sum()
        sell_count = (self.decisions == -1).sum()
        hold_count = (self.decisions == 0).sum()
        total_signals = buy_count + sell_count
        
        print(f"Total Buy Signals: {buy_count}")
        print(f"Total Sell Signals: {sell_count}")
        print(f"Total Hold Periods: {hold_count}")
        print(f"Signal Frequency: {total_signals / len(self.decisions) * 100:.2f}%")
        
        decisions_df = pd.DataFrame({
            'DateTime': self.price_data.index,
            'Decisions': self.decisions,
            'MidPrice': self.price_data['midPrice']
        })
        
        try:
            plt.figure(figsize=(15, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(decisions_df['DateTime'], decisions_df['MidPrice'], label='Mid Price', alpha=0.7)
            plt.title(f'Price and Trading Decisions - {self.strategy_type.upper()} Strategy')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            buy_points = decisions_df[decisions_df['Decisions'] == 1]
            sell_points = decisions_df[decisions_df['Decisions'] == -1]
            
            plt.scatter(buy_points['DateTime'], buy_points['MidPrice'], 
                        color='green', marker='^', s=50, label='Buy Signal', alpha=0.8)
            plt.scatter(sell_points['DateTime'], sell_points['MidPrice'], 
                        color='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
            plt.plot(decisions_df['DateTime'], decisions_df['MidPrice'], alpha=0.3, color='gray')
            plt.ylabel('Price')
            plt.xlabel('DateTime')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'graphs/trading_decisions_{self.strategy_type}.png')
            plt.show()
        except Exception as e:
            print(f"Warning: Could not create trading decisions plot for {self.strategy_type}: {e}")
            print("Continuing with analysis...")
        
        self.analysis_results = {
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'total_signals': total_signals,
            'signal_frequency': total_signals / len(self.decisions) * 100
        }
    
    def print_performance_report(self, stats: Dict, decisions: pd.DataFrame):
        portfolio = stats['portfolio']
        print("\n" + "="*60)
        print("📊 PORTFÖY PERFORMANS RAPORU (VectorBT)")
        print("="*60)
        
        print(f"💰 Toplam Getiri:           {stats['total_return']/100:.2%}")
        print(f"📈 Yıllık Getiri:           {stats['annualized_return']/100:.2%}")
        print(f"📊 Yıllık Volatilite:       {stats['annualized_volatility']/100:.2%}")
        print(f"⚖️  Sharpe Oranı:            {stats['sharpe_ratio']:.3f}")
        print(f"📉 Maksimum Drawdown:        {stats['max_drawdown']/100:.2%}")
        
        # Trading decisions istatistikleri ekle
        if decisions is not None:
            try:
                decisions_values = decisions.iloc[:, 0].values  # İlk kolonu al
                buy_signals = np.sum(decisions_values == 1)
                sell_signals = np.sum(decisions_values == -1)
                hold_signals = np.sum(decisions_values == 0)
                
                print(f"\n📊 TRADING SİNYALLERİ:")
                print(f"   🟢 Buy Sinyalleri:          {buy_signals}")
                print(f"   🔴 Sell Sinyalleri:         {sell_signals}")
                print(f"   ⚪ Hold Sinyalleri:         {hold_signals}")
                print(f"   📈 Buy/Sell Oranı:          {buy_signals/max(sell_signals,1):.2f}")
                
            except Exception as e:
                print(f"   ⚠️ Trading signals not available: {e}")
        
        if portfolio.trades is not None:
            try:
                trades = portfolio.trades.records
                
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

    def run_backtest(self):
        print(f"\n=== VectorBT Backtest for {self.strategy_type.upper()} Strategy ===")
        
        try:
            backtest_price = self.price_data['midPrice']
            decisions_df = pd.DataFrame(self.decisions)
            weights = decisions_df.div(decisions_df.abs().sum(axis=1), axis=0).fillna(0)
            
            pf = vbt.Portfolio.from_orders(
                close=backtest_price,
                size=weights,
                size_type='amount',
                freq='1min',
                init_cash=100,
                cash_sharing=True,
                call_seq='auto',
                fees=0.001,
                slippage=0.0005
            )
    
            full_stats = pf.stats()
            ann_factor = pf.returns().vbt.returns().ann_factor
            
            total_return = full_stats['Total Return [%]']
            annualized_return = (pf.returns().mean() * ann_factor) * 100
            annualized_volatility = pf.returns().std() * (ann_factor ** 0.5) * 100
            sharpe_ratio = full_stats['Sharpe Ratio']
            max_drawdown = full_stats['Max Drawdown [%]']
            
            stats_dict = {
                'portfolio': pf,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
            self.print_performance_report(stats_dict, decisions_df)
            
            print(f"\nAnn Factor:                         {ann_factor}")
            print("\nBacktest Stats:")
            print(f"Total Return [%]:                   {total_return:.3f}%")
            print(f"Annualized Expected Return [%]:     {annualized_return:.3f}%")
            print(f"Annualized Expected Volatility [%]: {annualized_volatility:.3f}%")
            print(f"Sharpe Ratio:                       {sharpe_ratio:.3f}")
            print(f"Max Drawdown [%]:                   {max_drawdown:.3f}%")
            print(f"Win Rate:                           {full_stats['Win Rate [%]']:.3f}%")
            print(f"Profit Factor:                      {full_stats['Profit Factor']:.3f}")
            print(f"Calmar Ratio:                       {full_stats['Calmar Ratio']:.3f}")
            
            self.create_backtest_plots(pf, self.strategy_type)
            
            return {
                'portfolio': pf,
                'stats': full_stats,
                'ann_factor': ann_factor
            }
            
        except Exception as e:
            print(f"Error in backtest for {self.strategy_type}: {e}")
            print("Attempting to create basic portfolio...")
            """
            try:
                backtest_price = df['midPrice']
                decisions_df = pd.DataFrame(decisions)
                weights = decisions_df.div(decisions_df.abs().sum(axis=1), axis=0).fillna(0)
                
                pf = vbt.Portfolio.from_orders(
                    close=backtest_price,
                    size=weights,
                    size_type='amount',
                    freq='1T',
                    init_cash=100,
                    cash_sharing=False,
                    call_seq='auto',
                    fees=0.001,
                    slippage=0.0005
                )
                
                full_stats = pf.stats()
                ann_factor = pf.returns().vbt.returns().ann_factor
                
                print(f"Fallback backtest successful!")
                print(f"Total Return [%]: {full_stats['Total Return [%]']:.3f}%")
                
                return {
                    'portfolio': pf,
                    'stats': full_stats,
                    'ann_factor': ann_factor
                }
                
            except Exception as e2:
                print(f"Fallback backtest also failed: {e2}")
                raise e2
            """

    def create_backtest_plots(self, pf, strategy_name):
        os.makedirs('graphs', exist_ok=True)
        
        try:
            plt.figure(figsize=(12, 6))
            pf.value().plot()
            plt.title(f'Portfolio Value - {strategy_name.upper()} Strategy')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'graphs/portfolio_value_{strategy_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(12, 6))
            returns = pf.returns()
            returns.hist(bins=50)
            plt.title(f'Returns Distribution - {strategy_name.upper()} Strategy')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'graphs/returns_distribution_{strategy_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
            """plt.figure(figsize=(12, 6))
            try:
                drawdown_plot = pf.plot_drawdown()
            except Exception as e:
                print(f"Warning: Could not create drawdown plot: {e}")
                plt.plot(pf.drawdown())
            plt.title(f'Drawdown Analysis - {strategy_name.upper()} Strategy')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'graphs/drawdown_{strategy_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(12, 6))
            try:
                underwater_plot = pf.plot_underwater()
            except Exception as e:
                print(f"Warning: Could not create underwater plot: {e}")
                plt.plot(pf.drawdown().cummin())
            plt.title(f'Underwater Plot - {strategy_name.upper()} Strategy')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'graphs/underwater_{strategy_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        """ 
        except Exception as e:
            print(f"Warning: Error creating plots for {strategy_name}: {e}")
            print("Continuing with backtest analysis...")
    