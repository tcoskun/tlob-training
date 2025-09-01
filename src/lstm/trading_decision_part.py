import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""
Trading Decision Part - Enhanced with Reduced Trading Frequency Options

This module implements various trading strategies with enhanced filtering mechanisms
to reduce the number of trading decisions across all strategies.

Key Features for Reducing Trading Frequency:

MOMENTUM STRATEGY:
1. Threshold-based filtering - Only trade when price changes exceed minimum threshold
2. Trend confirmation - Require consecutive signals in the same direction
3. Volatility filtering - Only trade when volatility is within acceptable range
4. Time-based filtering - Minimum intervals between trades
6. Profit taking and stop loss - Automatic position management

MEAN REVERSION STRATEGY:
1. Threshold-based filtering - Only trade when deviation from moving average exceeds minimum
2. Confirmation filtering - Require consecutive signals for confirmation
3. Time-based filtering - Minimum intervals between trades
5. Position management - Profit taking, stop loss, and maximum holding periods
6. Moving average window - Configurable window size for trend calculation

VOLATILITY BREAKOUT STRATEGY:
1. Breakout factor - Standard deviation multiplier (lower = fewer signals)
2. Confirmation required - Optional consecutive signal confirmation
3. Time-based filtering - Minimum intervals between trades
5. Position management - Profit taking and stop loss
6. Volatility window - Configurable window for volatility calculation

"""

def create_trading_decisions(price_data, strategy_type, y_hat_last, start_index):
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
    
    decisions = pd.Series(0, index=price_data.index, dtype=int)

    test_end_index = start_index + len(y_hat_last)
    if test_end_index > len(price_data):
        test_end_index = len(price_data)
        y_hat_last = y_hat_last[:len(price_data) - start_index]

    test_indices = price_data.index[start_index:test_end_index]

    if strategy_type == 'momentum':
        prev_prices = price_data['midPrice'].shift(1).loc[test_indices]
        
        price_changes = y_hat_last - prev_prices.values
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
        rolling_std = price_data['midPrice'].rolling(window=volatility_window).std()
        volatility_values = rolling_std.loc[test_indices]
        price_values = price_data['midPrice'].loc[test_indices]
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
            current_price = price_data['midPrice'].iloc[start_index + i]
            
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
        
        moving_avg = price_data['midPrice'].rolling(window=ma_window).mean()
        ma_values = moving_avg.loc[test_indices]
        predicted_prices = y_hat_last
        
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
            current_price = price_data['midPrice'].iloc[start_index + i]
            
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
        
        moving_avg = price_data['midPrice'].rolling(window=volatility_window).mean()
        std_dev = price_data['midPrice'].rolling(window=volatility_window).std()
        
        upper_band = moving_avg + std_dev * breakout_factor
        lower_band = moving_avg - std_dev * breakout_factor
        
        upper_values = upper_band.loc[test_indices]
        lower_values = lower_band.loc[test_indices]
        predicted_prices = y_hat_last
        
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
            current_price = price_data['midPrice'].iloc[start_index + i]
            
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
    
    return decisions

def adjust_momentum_trading_frequency(
    min_threshold=0.0005,
    trend_window=3,
    volatility_max=0.01,
    volatility_window=10,
    min_trade_interval=5,
    profit_take_threshold=0.002,
    stop_loss_threshold=0.001
):
    """
    Helper function to easily adjust momentum strategy parameters for reducing trading frequency.
    
    Parameters:
    - min_threshold: Minimum price change percentage to trigger a trade (higher = fewer trades)
    - trend_window: Number of consecutive signals needed for trend confirmation (higher = fewer trades)
    - volatility_max: Maximum volatility allowed for trading (lower = fewer trades)
    - volatility_window: Window size for volatility calculation
    - min_trade_interval: Minimum periods between trades (higher = fewer trades)
    - profit_take_threshold: Profit taking threshold (higher = longer holding periods)
    - stop_loss_threshold: Stop loss threshold (lower = longer holding periods)
    
    Returns:
    - Dictionary with updated configuration
    """
    global MOMENTUM_CONFIG
    
    MOMENTUM_CONFIG = {
        'min_threshold': min_threshold,
        'trend_window': trend_window,
        'volatility_max': volatility_max,
        'volatility_window': volatility_window,
        'min_trade_interval': min_trade_interval,
        'profit_take_threshold': profit_take_threshold,
        'stop_loss_threshold': stop_loss_threshold
    }
    
    print("Momentum strategy parameters updated:")
    for key, value in MOMENTUM_CONFIG.items():
        print(f"  {key}: {value}")
    
    return MOMENTUM_CONFIG

def adjust_mean_reversion_trading_frequency(
    min_threshold=0.0005,
    ma_window=20,
    confirmation_window=3,
    min_trade_interval=8,
    profit_take_threshold=0.003,
    stop_loss_threshold=0.002,
    max_holding_periods=50
):
    """
    Helper function to easily adjust mean reversion strategy parameters for reducing trading frequency.
    
    Parameters:
    - min_threshold: Minimum deviation from moving average (higher = fewer trades)
    - ma_window: Moving average window size
    - confirmation_window: Consecutive signals needed for confirmation (higher = fewer trades)
    - min_trade_interval: Minimum periods between trades (higher = fewer trades)
    - profit_take_threshold: Profit taking threshold (higher = longer holding periods)
    - stop_loss_threshold: Stop loss threshold (lower = longer holding periods)
    - max_holding_periods: Maximum periods to hold a position
    
    Returns:
    - Dictionary with updated configuration
    """
    global MEAN_REVERSION_CONFIG
    
    MEAN_REVERSION_CONFIG = {
        'min_threshold': min_threshold,
        'ma_window': ma_window,
        'confirmation_window': confirmation_window,
        'min_trade_interval': min_trade_interval,
        'profit_take_threshold': profit_take_threshold,
        'stop_loss_threshold': stop_loss_threshold,
        'max_holding_periods': max_holding_periods
    }
    
    print("Mean reversion strategy parameters updated:")
    for key, value in MEAN_REVERSION_CONFIG.items():
        print(f"  {key}: {value}")
    
    return MEAN_REVERSION_CONFIG

def adjust_volatility_breakout_trading_frequency(
    volatility_window=20,
    breakout_factor=8,
    min_trade_interval=10,
    profit_take_threshold=0.004,
    stop_loss_threshold=0.002,
    confirmation_required=True,
    confirmation_window=2
):
    """
    Helper function to easily adjust volatility breakout strategy parameters for reducing trading frequency.
    
    Parameters:
    - volatility_window: Window for volatility calculation
    - breakout_factor: Standard deviation multiplier (lower = fewer signals)
    - min_trade_interval: Minimum periods between trades (higher = fewer trades)
    - profit_take_threshold: Profit taking threshold (higher = longer holding periods)
    - stop_loss_threshold: Stop loss threshold (lower = longer holding periods)
    - confirmation_required: Whether to require confirmation before trading
    - confirmation_window: Number of periods for confirmation (higher = fewer trades)
    
    Returns:
    - Dictionary with updated configuration
    """
    global VOLATILITY_BREAKOUT_CONFIG
    
    VOLATILITY_BREAKOUT_CONFIG = {
        'volatility_window': volatility_window,
        'breakout_factor': breakout_factor,
        'min_trade_interval': min_trade_interval,
        'profit_take_threshold': profit_take_threshold,
        'stop_loss_threshold': stop_loss_threshold,
        'confirmation_required': confirmation_required,
        'confirmation_window': confirmation_window
    }
    
    print("Volatility breakout strategy parameters updated:")
    for key, value in VOLATILITY_BREAKOUT_CONFIG.items():
        print(f"  {key}: {value}")
    
    return VOLATILITY_BREAKOUT_CONFIG

def analyze_decisions(decisions, price_data, strategy_type):
    print(f"\n=== Trading Decision Analysis for {strategy_type.upper()} Strategy ===")
    
    buy_count = (decisions == 1).sum()
    sell_count = (decisions == -1).sum()
    hold_count = (decisions == 0).sum()
    total_signals = buy_count + sell_count
    
    print(f"Total Buy Signals: {buy_count}")
    print(f"Total Sell Signals: {sell_count}")
    print(f"Total Hold Periods: {hold_count}")
    print(f"Signal Frequency: {total_signals / len(decisions) * 100:.2f}%")
    
    decisions_df = pd.DataFrame({
        'DateTime': price_data.index,
        'Decisions': decisions,
        'MidPrice': price_data['midPrice']
    })
    
    try:
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(decisions_df['DateTime'], decisions_df['MidPrice'], label='Mid Price', alpha=0.7)
        plt.title(f'Price and Trading Decisions - {strategy_type.upper()} Strategy')
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
        plt.savefig(f'graphs/trading_decisions_{strategy_type}.png')
        plt.show()
    except Exception as e:
        print(f"Warning: Could not create trading decisions plot for {strategy_type}: {e}")
        print("Continuing with analysis...")
    
    return {
        'buy_count': buy_count,
        'sell_count': sell_count,
        'hold_count': hold_count,
        'total_signals': total_signals,
        'signal_frequency': total_signals / len(decisions) * 100
    }

def main():
    print("=== Trading Decision Part ===")
    
    print("\n--- Adjusting All Strategy Parameters ---")
    
    print("\n1. MOMENTUM STRATEGY:")
    print("Current default parameters (moderate trading frequency):")
    adjust_momentum_trading_frequency()
    
    print("\n2. MEAN REVERSION STRATEGY:")
    print("Current default parameters (moderate trading frequency):")
    adjust_mean_reversion_trading_frequency()
    
    print("\n3. VOLATILITY BREAKOUT STRATEGY:")
    print("Current default parameters (moderate trading frequency):")
    adjust_volatility_breakout_trading_frequency()
    
    print("\n--- Example: Conservative settings for all strategies (fewer trades) ---")
    
    adjust_momentum_trading_frequency(
        min_threshold=0.001,        # Higher threshold = fewer trades
        trend_window=5,             # More confirmations needed = fewer trades
        volatility_max=0.005,       # Lower volatility allowed = fewer trades
        min_trade_interval=15,      # Longer intervals = fewer trades
        profit_take_threshold=0.005, # Higher profit target = longer holding
        stop_loss_threshold=0.002   # Higher stop loss = longer holding
    )
    
    adjust_mean_reversion_trading_frequency(
        min_threshold=0.001,        # Higher threshold = fewer trades
        confirmation_window=5,      # More confirmations needed = fewer trades
        min_trade_interval=15,      # Longer intervals = fewer trades
        profit_take_threshold=0.005, # Higher profit target = longer holding
        stop_loss_threshold=0.003,  # Higher stop loss = longer holding
        max_holding_periods=100     # Longer holding periods
    )
    
    adjust_volatility_breakout_trading_frequency(
        breakout_factor=12,         # Higher factor = fewer signals
        min_trade_interval=20,      # Longer intervals = fewer trades
        profit_take_threshold=0.006, # Higher profit target = longer holding
        stop_loss_threshold=0.003,  # Higher stop loss = longer holding
        confirmation_required=True, # Require confirmation
        confirmation_window=3       # More confirmations needed
    )
    
    print("\n--- Resetting to default parameters for analysis ---")
    adjust_momentum_trading_frequency()
    adjust_mean_reversion_trading_frequency()
    adjust_volatility_breakout_trading_frequency()
    
    try:
        with open('models/tensorflow_results.pkl', 'rb') as f:
            tensorflow_results = pickle.load(f)
        print("Successfully loaded TensorFlow results")
    except FileNotFoundError:
        print("Error: 'models/tensorflow_results.pkl' not found. Please run tensorflow_part.py first.")
        return
    except Exception as e:
        print(f"Error loading TensorFlow results: {e}")
        return
    
    df = tensorflow_results['df']
    y_hat_last = tensorflow_results['y_hat_last']
    SEQ_LEN = tensorflow_results['SEQ_LEN']
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Predicted prices length: {len(y_hat_last)}")
    print(f"Sequence length: {SEQ_LEN}")
    
    start_index = SEQ_LEN + int(0.5 * (len(df) - SEQ_LEN))
    print(f"Start index for validation+test period: {start_index}")
    print(f"Split: 50% training, 50% validation+testing")
    
    strategies = ['momentum', 'mean_reversion', 'volatility_breakout']
    all_results = {}
    
    for strategy_type in strategies:
        print(f"\n--- Creating trading decisions for {strategy_type.upper()} strategy ---")
        
        try:
            decisions = create_trading_decisions(
                df, 
                strategy_type, 
                y_hat_last, 
                start_index
            )
            
            analysis_results = analyze_decisions(decisions, df, strategy_type)
            all_results[strategy_type] = {
                'decisions': decisions,
                'analysis': analysis_results
            }
            
            print(f"Trading decisions created for {strategy_type} strategy")
            
        except Exception as e:
            print(f"Error creating trading decisions for {strategy_type}: {e}")
            print(f"Continuing with other strategies...")
            continue
    
    if not all_results:
        print("No successful trading strategies created. Cannot proceed.")
        return
    
    trading_results = {
        'df': df,
        'strategies': all_results,
        'start_index': start_index,
        'SEQ_LEN': SEQ_LEN
    }
    
    with open('trading_results.pkl', 'wb') as f:
        pickle.dump(trading_results, f)
    
    print("\nTrading decisions saved to 'trading_results.pkl'")
    print("Trading decision part completed successfully!")
    
    print("\n=== Summary ===")
    for strategy, results in all_results.items():
        analysis = results['analysis']
        print(f"{strategy.upper()}: {analysis['buy_count']} buys, {analysis['sell_count']} sells, "
              f"{analysis['signal_frequency']:.2f}% signal frequency")

if __name__ == "__main__":
    main()