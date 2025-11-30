import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

# TensorFlow/Keras for deep learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping

# Wavelet Transform for advanced noise removal
import pywt

# ---------------------- ADVANCED NOISE REMOVAL FUNCTIONS ---------------------- #

def wavelet_denoise(series, wavelet='db1'):
    coeffs = pywt.wavedec(series, wavelet, mode='smooth')
    sigma_est = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma_est * np.sqrt(2 * np.log(len(series)))
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    denoised_series = pywt.waverec(new_coeffs, wavelet, mode='smooth')
    denoised_series = denoised_series[:len(series)]
    return denoised_series

def kalman_denoise(series):
    try:
        from pykalman import KalmanFilter
    except ImportError:
        st.error("Please install pykalman for Kalman filtering: pip install pykalman")
        return series
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series[0],
        n_dim_obs=1
    )
    state_means, _ = kf.filter(series)
    return state_means.flatten()

def moving_average_smoothing(series, window_size=5):
    # Ensure series is 1D by flattening before applying rolling mean
    return pd.Series(series.flatten()).rolling(window=window_size, min_periods=1).mean().values

# ---------------------- MODEL TRAINING FUNCTIONS (Global) ---------------------- #

def create_dataset(series, look_back=10):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i : i + look_back])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

def train_lstm(X_tr, y_tr, X_val, y_val, epochs=20, batch_size=32):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_tr.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])
    return model

def train_gru(X_tr, y_tr, X_val, y_val, epochs=20, batch_size=32):
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=(X_tr.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])
    return model

def train_tcn(X_tr, y_tr, X_val, y_val, epochs=20, batch_size=32):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_tr.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])
    return model

# ---------------------- STREAMLIT APP ---------------------- #

st.title("Intraday Stock Prediction with Advanced Noise Removal & Multiple Models")

# ---------- Sidebar Inputs ----------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
train_start = st.sidebar.text_input("Training Start Date (YYYY-MM-DD)", value="2025-01-15")
train_end = st.sidebar.text_input("Training End Date (YYYY-MM-DD)", value="2025-03-11")
test_start = st.sidebar.text_input("Test Start Date (YYYY-MM-DD)", value="2025-03-12")
test_end = st.sidebar.text_input("Test End Date (YYYY-MM-DD)", value="2025-03-13")

# Choose advanced noise removal method
noise_method = st.sidebar.radio(
    "Choose Advanced Noise Removal Method",
    ("None", "Wavelet", "Kalman", "Moving Average"),
    index=1
)

# ----- MAIN BUTTON: Load Data & Train Models -----
if st.sidebar.button("Load Data & Train Models"):
    st.write("Fetching training data...")
    train_data = yf.download(ticker, start=train_start, end=train_end, interval="5m")
    if train_data.empty:
        st.error("No training data fetched. Please check the ticker or date range.")
        st.stop()
    st.write("Training Data (first 5 rows):")
    st.dataframe(train_data.head())

    st.write("Fetching test data...")
    test_data = yf.download(ticker, start=test_start, end=test_end, interval="5m")
    if test_data.empty:
        st.error("No test data fetched. Please check the ticker or date range.")
        st.stop()
    st.write("Test Data (first 5 rows):")
    st.dataframe(test_data.head())

    train_series = train_data['Close'].values
    test_series = test_data['Close'].values

    # ---------- Advanced Noise Removal ----------
    if noise_method == "Wavelet":
        st.write("Applying Wavelet Denoising...")
        train_series = wavelet_denoise(train_series, wavelet='db1')
        test_series = wavelet_denoise(test_series, wavelet='db1')
    elif noise_method == "Kalman":
        st.write("Applying Kalman Filter Denoising...")
        train_series = kalman_denoise(train_series)
        test_series = kalman_denoise(test_series)
    elif noise_method == "Moving Average":
        st.write("Applying Moving Average Smoothing...")
        train_series = moving_average_smoothing(train_series)
        test_series = moving_average_smoothing(test_series)
    else:
        st.write("No advanced noise removal applied.")

    train_series = np.array(train_series).flatten()
    test_series = np.array(test_series).flatten()

    train_series = pd.Series(train_series, index=train_data.index)
    test_series = pd.Series(test_series, index=test_data.index)

    # ---------- Create Dataset for Time Series Prediction ----------
    look_back = 10
    X_train, y_train = create_dataset(train_series.values, look_back)
    X_test, y_test = create_dataset(test_series.values, look_back)

    if X_train.size == 0 or X_test.size == 0:
        st.error("Not enough data to create the dataset. Adjust the date range or look_back parameter.")
        st.stop()

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    split_idx = int(len(X_train) * 0.8)
    X_tr, y_tr = X_train[:split_idx], y_train[:split_idx]
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]

    # ---------- Model Training ----------
    st.write("Training models (this may take a while)...")
    with st.spinner("Training LSTM..."):
        lstm_model = train_lstm(X_tr, y_tr, X_val, y_val)
    with st.spinner("Training GRU..."):
        gru_model = train_gru(X_tr, y_tr, X_val, y_val)
    with st.spinner("Training TCN..."):
        tcn_model = train_tcn(X_tr, y_tr, X_val, y_val)

    # ---------- Predictions on Test Data ----------
    lstm_pred = lstm_model.predict(X_test)
    gru_pred = gru_model.predict(X_test)
    tcn_pred = tcn_model.predict(X_test)

    # ---------- Metrics Calculation ----------
    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred) * 100
        return rmse, r2

    rmse_lstm, r2_lstm = calc_metrics(y_test, lstm_pred)
    rmse_gru, r2_gru = calc_metrics(y_test, gru_pred)
    rmse_tcn, r2_tcn = calc_metrics(y_test, tcn_pred)

    # Create a time axis for plotting (drop first look_back timestamps)
    time_axis = test_data.index[look_back:]
    actual_values = y_test

    # ---------- Show Line Charts for Each Model ----------
    st.subheader("LSTM: Actual vs. Predicted")
    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(x=time_axis, y=actual_values, mode='lines', name='Actual'))
    fig_lstm.add_trace(go.Scatter(x=time_axis, y=lstm_pred.flatten(), mode='lines', name='Predicted'))
    fig_lstm.update_layout(
        title=f"LSTM Predictions (RMSE={rmse_lstm:.4f}, R²={r2_lstm:.2f}%)",
        xaxis_title="Time", yaxis_title="Price"
    )
    st.plotly_chart(fig_lstm, use_container_width=True)

    st.subheader("GRU: Actual vs. Predicted")
    fig_gru = go.Figure()
    fig_gru.add_trace(go.Scatter(x=time_axis, y=actual_values, mode='lines', name='Actual'))
    fig_gru.add_trace(go.Scatter(x=time_axis, y=gru_pred.flatten(), mode='lines', name='Predicted'))
    fig_gru.update_layout(
        title=f"GRU Predictions (RMSE={rmse_gru:.4f}, R²={r2_gru:.2f}%)",
        xaxis_title="Time", yaxis_title="Price"
    )
    st.plotly_chart(fig_gru, use_container_width=True)

    st.subheader("TCN: Actual vs. Predicted")
    fig_tcn = go.Figure()
    fig_tcn.add_trace(go.Scatter(x=time_axis, y=actual_values, mode='lines', name='Actual'))
    fig_tcn.add_trace(go.Scatter(x=time_axis, y=tcn_pred.flatten(), mode='lines', name='Predicted'))
    fig_tcn.update_layout(
        title=f"TCN Predictions (RMSE={rmse_tcn:.4f}, R²={r2_tcn:.2f}%)",
        xaxis_title="Time", yaxis_title="Price"
    )
    st.plotly_chart(fig_tcn, use_container_width=True)

    # ---------- Display Model Metrics ----------
    st.subheader("Model Metrics")
    metrics_df = pd.DataFrame({
        "Model": ["LSTM", "GRU", "TCN"],
        "RMSE": [rmse_lstm, rmse_gru, rmse_tcn],
        "R² Accuracy (%)": [r2_lstm, r2_gru, r2_tcn]
    })
    st.table(metrics_df)
    
    # ---- Store variables in session_state for future use ----
    st.session_state['model_metrics'] = {"LSTM": rmse_lstm, "GRU": rmse_gru, "TCN": rmse_tcn}
    best_model_name = min(st.session_state['model_metrics'], key=st.session_state['model_metrics'].get)
    st.session_state['best_model_name'] = best_model_name
    st.session_state['X_train'] = X_train
    st.session_state['y_train'] = y_train
    st.session_state['noise_method'] = noise_method
    st.session_state['look_back'] = look_back
    st.session_state['ticker'] = ticker
    st.session_state['train_start'] = train_start
    st.session_state['test_start'] = test_start
    st.session_state['test_end'] = test_end
    # Initialize create_dataset function in session_state if not already set
    if 'create_dataset' not in st.session_state:
        st.session_state['create_dataset'] = create_dataset

# ---- New Functionality: Best Model Retraining & Future Prediction ----
if 'best_model_name' in st.session_state:
    st.write(f"**The model with the lowest RMSE is {st.session_state['best_model_name']}. Should we predict the next day's closing prices?**")
    if st.button(f"Predict Next Day with {st.session_state['best_model_name']}"):
        best_model_name = st.session_state['best_model_name']
        st.write(f"Retraining {best_model_name} on data from {st.session_state['train_start']} to {st.session_state['test_start']}...")
        
        # Ensure the 'create_dataset' function is available in session_state
        if 'create_dataset' not in st.session_state:
            st.session_state['create_dataset'] = create_dataset

        # Fetch extended training data from Training Start Date to Test Start Date
        extended_data = yf.download(st.session_state['ticker'], start=st.session_state['train_start'], end=st.session_state['test_start'], interval="5m")
        if extended_data.empty:
            st.error("No extended training data fetched. Please check the date range.")
            st.stop()
        
        extended_series = extended_data['Close'].values
        if st.session_state['noise_method'] == "Wavelet":
            extended_series = wavelet_denoise(extended_series, wavelet='db1')
        elif st.session_state['noise_method'] == "Kalman":
            extended_series = kalman_denoise(extended_series)
        elif st.session_state['noise_method'] == "Moving Average":
            extended_series = moving_average_smoothing(extended_series)
        extended_series = np.array(extended_series).flatten()
        extended_series = pd.Series(extended_series, index=extended_data.index)
        
        # Create dataset for extended training data
        create_dataset_fn = st.session_state['create_dataset']
        X_extended, y_extended = create_dataset_fn(extended_series.values, st.session_state['look_back'])
        X_extended = X_extended.reshape((X_extended.shape[0], X_extended.shape[1], 1))
        
        # Retrain the best model on extended data:
        if best_model_name == "LSTM":
            best_model = train_lstm(X_extended, y_extended, X_extended, y_extended)
        elif best_model_name == "GRU":
            best_model = train_gru(X_extended, y_extended, X_extended, y_extended)
        elif best_model_name == "TCN":
            best_model = train_tcn(X_extended, y_extended, X_extended, y_extended)
        
        # Fetch next day's data starting from Test End Date at 5-minute intervals
        next_day = (pd.to_datetime(st.session_state['test_end']) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        future_data = yf.download(st.session_state['ticker'], start=st.session_state['test_end'], end=next_day, interval="5m")
        if future_data.empty:
            st.error("No future data fetched. Please check the test end date.")
            st.stop()
        
        # Filter future data to market hours based on ticker suffix:
        if st.session_state['ticker'].endswith('.NS') or st.session_state['ticker'].endswith('.BO'):
            market_open = "09:15"
            market_close = "15:30"
        else:
            market_open = "09:30"
            market_close = "16:00"
        future_data = future_data.between_time(market_open, market_close)
        
        # Create dataset for future data using the "Close" column
        create_dataset_fn = st.session_state['create_dataset']
        future_series = future_data['Close'].values
        X_future, _ = create_dataset_fn(future_series, st.session_state['look_back'])
        X_future = X_future.reshape((X_future.shape[0], X_future.shape[1], 1))
        future_predictions = best_model.predict(X_future)
        
        # Determine prediction length and corresponding datetime array
        pred_length = len(future_predictions)
        pred_dates = list(future_data.index[st.session_state['look_back']:st.session_state['look_back']+pred_length])
        
        # Prepare results DataFrame with exactly 3 columns
        results_df = pd.DataFrame({
            "Datetime": pred_dates,
            "Predicted Closing Price": future_predictions.flatten(),
            "High Price": future_data['High'].values[st.session_state['look_back']:st.session_state['look_back']+pred_length].flatten()
        })
        st.write("### Next Day Predictions")
        st.dataframe(results_df)
        
        # Plot future predictions (only predicted closing price)
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=results_df["Datetime"], y=results_df["Predicted Closing Price"], mode='lines', name='Predicted Close'))
        st.plotly_chart(fig_future, use_container_width=True)
