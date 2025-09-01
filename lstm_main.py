#scikit learn mapei kullanmak (sklearn)  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
#tensorflow mapei kullanmak (tf)  https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanAbsolutePercentageError
import os
import vectorbt as vbt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout, Activation, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.array_wrapper['freq'] = '1T'

def create_trading_decisions(price_data, strategy_type, y_hat_last, start_index):
    decisions = pd.Series(0, index=price_data.index, dtype=int)
    
    if strategy_type == 'momentum':
        test_end_index = start_index + len(y_hat_last)
        if test_end_index > len(price_data):
            test_end_index = len(price_data)
            y_hat_last = y_hat_last[:len(price_data) - start_index]
        
        test_indices = price_data.index[start_index:test_end_index]
        
        prev_prices = price_data['midPrice'].shift(1).loc[test_indices]
        
        buy_signals = y_hat_last > prev_prices.values
        sell_signals = y_hat_last < prev_prices.values
        
        decisions.loc[test_indices] = np.select(
            [buy_signals, sell_signals],
            [1, -1],
            default=0
        )
    
    elif strategy_type == 'mean_reversion':
        test_end_index = start_index + len(y_hat_last)
        if test_end_index > len(price_data):
            test_end_index = len(price_data)
            y_hat_last = y_hat_last[:len(price_data) - start_index]
        
        test_indices = price_data.index[start_index:test_end_index]
        
        window_size = 20
        moving_avg = price_data['midPrice'].rolling(window=window_size).mean()
        
        ma_values = moving_avg.loc[test_indices]
        predicted_prices = y_hat_last
        
        threshold = 0.001  # 0.1% threshold
        
        buy_signals = (predicted_prices < ma_values.values) & (ma_values.values - predicted_prices > threshold * ma_values.values)
        sell_signals = (predicted_prices > ma_values.values) & (predicted_prices - ma_values.values > threshold * ma_values.values)
        
        decisions.loc[test_indices] = np.select(
            [buy_signals, sell_signals],
            [1, -1],
            default=0
        )
    
    return decisions

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    epsilon = 1e-4
    y_true_safe = tf.clip_by_value(y_true, epsilon, float('inf'))
    
    percentage_errors = tf.abs((y_true_safe - y_pred) / y_true_safe) * 100
    
    max_percentage = 1000.0
    percentage_errors_clipped = tf.clip_by_value(percentage_errors, 0.0, max_percentage)
    
    return tf.reduce_mean(percentage_errors_clipped)

def numpy_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    print(f"MAPE Debug - y_true range: {np.min(y_true):.6f} to {np.max(y_true):.6f}")
    print(f"MAPE Debug - y_pred range: {np.min(y_pred):.6f} to {np.max(y_pred):.6f}")
    print(f"MAPE Debug - y_true mean: {np.mean(y_true):.6f}")
    print(f"MAPE Debug - y_pred mean: {np.mean(y_pred):.6f}")
    
    epsilon = 1e-6
    y_true_safe = np.clip(y_true, epsilon, None)
    
    percentage_errors = np.abs((y_true_safe - y_pred) / y_true_safe) * 100
    
    max_percentage = 1000.0  # Cap at 1000% to prevent extreme values
    percentage_errors_clipped = np.clip(percentage_errors, 0.0, max_percentage)
    
    print(f"MAPE Debug - Percentage errors range: {np.min(percentage_errors_clipped):.2f}% to {np.max(percentage_errors_clipped):.2f}%")
    print(f"MAPE Debug - Percentage errors mean: {np.mean(percentage_errors_clipped):.2f}%")
    print(f"MAPE Debug - Number of clipped values: {np.sum(percentage_errors > max_percentage)}")
    
    return np.mean(percentage_errors_clipped)

def to_sequences(data, seq_len):
    """
    Converts a 2D array into sequences of a specified length.
    Each sequence will have `seq_len` time steps.
    """
    d = []
    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])
    return np.array(d)

def preprocess(data_raw, seq_len, train_split):
    """
    Preprocesses the raw data into sequences and splits it into
    training and testing sets.
    
    Args:
        data_raw (np.array): The raw input data.
        seq_len (int): The length of each sequence.
        train_split (float): The proportion of data to use for training.
        
    Returns:
        tuple: X_train, y_train, X_test, y_test
    """
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])

    # X will be sequences of length (SEQ_LEN - 10)
    # y will be the 10th step after the X sequence ends
    X_train = data[:num_train, :-10, :]
    y_train = data[:num_train, -10, :]

    X_test = data[num_train:, :-10, :]
    y_test = data[num_train:, -10, :]

    return X_train, y_train, X_test, y_test

def run_backtest(df, decisions):
    print("\n VectorBT backtest...")
    
    backtest_price = df['midPrice']
    decisions_df = pd.DataFrame(decisions)
    weights = decisions_df.div(decisions_df.abs().sum(axis=1), axis=0).fillna(0)
    
    pf = vbt.Portfolio.from_orders(
        close = backtest_price,
        size=weights,
        size_type='amount',
        freq='1T',
        init_cash=100,
        cash_sharing=True,
        call_seq='auto',
        fees=0.001,
        slippage=0.0005
    )

    full_stats = pf.stats()
    ann_factor = pf.returns().vbt.returns().ann_factor
    print(f"Ann Factor:                         {ann_factor}")
    print("\nBacktest Stats:")
    print(f"Ann Factor:                         {ann_factor}")
    print(f"Total Return [%]:                   {full_stats['Total Return [%]']:.3f}%")
    print(f"Annualized Expected Return [%]:     {(pf.returns().mean() * ann_factor):.3f}%")
    print(f"Annualized Expected Volatility [%]: {pf.returns().std() * (ann_factor ** .5):.3f}%")
    print(f"Sharpe Ratio:                       {full_stats['Sharpe Ratio']:.3f}")
    print(f"Sharpe Ratio:                       {((pf.returns().mean() * ann_factor)/(pf.returns().std() * (ann_factor ** .5))):.3f}")
    print(f"Max Drawdown [%]:                   {full_stats['Max Drawdown [%]']:.3f}%")

    pf.value().plot()
    plt.title('Portfolio Value')
    plt.show()

    print('Values', pf.value())
    print('Returns', pf.returns())

    print("\nPortfolio Plot:")
    try:
        fig = pf.plot()
    except Exception as e:
        print(f"Warning: Could not create portfolio plot: {e}")
        # Create a simple portfolio plot manually
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(pf.value())
        plt.title('Portfolio Value')
        plt.subplot(2, 1, 2)
        plt.plot(pf.returns())
        plt.title('Portfolio Returns')
        fig = plt.gcf()
    fig.show()

    print("\nVectorBT backtest completed.")

def main():
    """
    Main function to run the LSTM model training and evaluation.
    """
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED) 

    """gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nTensorFlow detected {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("\nNo GPU .") """

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    mps_device_name = "/device:GPU:0"
    if tf.config.list_physical_devices('GPU'):
        print("\n detected GPU.")
        try:
            tf.config.set_logical_device_configuration(
                tf.config.list_physical_devices('GPU')[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
            )
            print("using GPU .")
        except RuntimeError as e:
            print(e)
        
        device_name = mps_device_name
    else:
        print("\nNo GPU .")
        device_name = "/device:CPU:0"
        
    print(f"Using device: {device_name}")  
 
    # Read all CSV files from data directory in alphabetical order
    data_dir = "data"
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Read and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        print(f"Reading {csv_file}...")
        temp_df = pd.read_csv(csv_path, sep=';', parse_dates=['DateTime'])
        dfs.append(temp_df)
        print(f"  - Shape: {temp_df.shape}")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined DataFrame shape: {df.shape}")
    
    df = df.sort_values('DateTime').reset_index(drop=True)
    print(f"DataFrame sorted by DateTime, final shape: {df.shape}")

    df['midPrice'] = (df['Level 1 Bid Price'] + df['Level 1 Ask Price']) / 2
    
    feature_columns = [
        'Depth Ratio',
        'Last Price', 
        'Total Bid Volume',
        ' Total Ask Volume',
        'Level 1 Bid Price',
        'Level 1 Bid Volume',
        'Level 1 Ask Price', 
        'Level 1 Ask Volume',
        'Level 2 Bid Price',
        'Level 2 Bid Volume',
        'Level 2 Ask Price', 
        'Level 2 Ask Volume',
        'Level 3 Bid Price',
        'Level 3 Bid Volume',
        'Level 3 Ask Price', 
        'Level 3 Ask Volume',
        'Level 4 Bid Price',
        'Level 4 Bid Volume',
        'Level 4 Ask Price', 
        'Level 4 Ask Volume',
        'Level 5 Bid Price',
        'Level 5 Bid Volume',
        'Level 5 Ask Price', 
        'Level 5 Ask Volume',
        'midPrice'
    ]
    
    target_column = 'midPrice'
    
    print(f"\nSelected Features: {feature_columns}")
    print(f"Target Column: {target_column}")
    
    feature_data = df[feature_columns].values
    
    if np.isnan(feature_data).any():
        print("Warning: NaN values found in feature data. Forward-filling them.")
        feature_data = pd.DataFrame(feature_data, columns=feature_columns).fillna(method='ffill').values
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(feature_data)
    
    SEQ_LEN = 300
    X_train, y_train, X_test, y_test = preprocess(scaled_features, SEQ_LEN, train_split = 0.9)
    
    DROPOUT = 0.2
    WINDOW_SIZE = SEQ_LEN - 10
    N_FEATURES = scaled_features.shape[1]

    model = keras.Sequential()
    model.add(LSTM(WINDOW_SIZE, return_sequences=True, input_shape=(WINDOW_SIZE, N_FEATURES)))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(WINDOW_SIZE, return_sequences=True))
    model.add(Dropout(rate=DROPOUT))
    model.add(LSTM(WINDOW_SIZE, return_sequences=False)) 
    model.add(Dropout(rate=DROPOUT))
    model.add(Dense(units=N_FEATURES))
    model.add(Activation('linear'))

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-3
    )
    
    checkpoint_filepath = 'models/best_lstm_model.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    adam = Adam(learning_rate=1e-4)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=[mean_absolute_percentage_error])
    
    BATCH_SIZE = 500

    print("\nStarting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=350,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        validation_split=0.2,
        callbacks=[lr_scheduler, model_checkpoint_callback]
    )
    print("Model training finished.")
    
    if os.path.exists(checkpoint_filepath):
        print(f"\nLoading best model from {checkpoint_filepath} for evaluation.")
        best_model = keras.models.load_model(checkpoint_filepath, 
                                            custom_objects={'mean_absolute_percentage_error': mean_absolute_percentage_error})
    else:
        print(f"\nError: Best model not found at {checkpoint_filepath}. Using the last trained model.")
        best_model = model 

    print("\nEvaluating model on test data...")
    test_loss, test_mape = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (best model): {test_loss:.6f}")
    print(f"Test MAPE (best model): {test_mape:.2f}%")
    
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nMaking predictions on test data...")
    y_hat = best_model.predict(X_test)

    y_test_inverse = scaler.inverse_transform(y_test)
    y_hat_inverse = scaler.inverse_transform(y_hat)
    
    last_col_idx = feature_columns.index('midPrice') 
    y_test_last = y_test_inverse[:, last_col_idx]
    y_hat_last = y_hat_inverse[:, last_col_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_last, label="Actual Mid Price", color='green', alpha=0.7)
    plt.plot(y_hat_last, label="Predicted Mid Price", color='red', alpha=0.7)

    plt.title('Mid Price Prediction - Multivariate LSTM')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    mse = np.mean((y_test_last - y_hat_last) ** 2)
    mae = np.mean(np.abs(y_test_last - y_hat_last))
    rmse = np.sqrt(mse)
    mape = numpy_mape(y_test_last, y_hat_last)

    print(f"\nPrediction Metrics (on inverse transformed 'midPrice'):")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.2f}%")
    
    print(f"\nFeature columns used for training: {feature_columns}")
    print(f"Number of features: {N_FEATURES}")

    print("\nMaking predictions on test data...")
    y_hat = best_model.predict(X_test)

    y_test_inverse = scaler.inverse_transform(y_test)
    y_hat_inverse = scaler.inverse_transform(y_hat)
    
    last_col_idx = feature_columns.index('midPrice') 
    y_test_last = y_test_inverse[:, last_col_idx]
    y_hat_last = y_hat_inverse[:, last_col_idx]

    strategy_type = 'mean_reversion'
    
    decisions = create_trading_decisions(
        df, 
        strategy_type, 
        y_hat_last, 
        start_index=SEQ_LEN + int(0.9 * (len(df) - SEQ_LEN))
    )
    """print(decisions.head(10))
    print("--------------------------------- \n ")
    print(decisions.tail(10))"""
    print("\n VectorBT backtest ...")

    run_backtest(df, decisions)

if __name__ == "__main__":
    main()