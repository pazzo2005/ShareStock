import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime as dt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Title of the app
st.title("Welcome to the Stock Dash App!")

# User input for stock code
stock_code = st.text_input("Enter Stock Code")

# Date range picker
start_date = st.date_input("Start Date", value=dt(2021, 1, 1))
end_date = st.date_input("End Date", value=dt.now())

# Input for forecast days
forecast_days = st.number_input("Days of Forecast", min_value=1)

def train_svr_model(df):
    X = df[['Open', 'High', 'Low', 'Close']].values
    y = df['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)

    predictions = svr.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return svr, mse, mae

def predict_future_prices(model, df, forecast_days):
    last_data = df[['Open', 'High', 'Low', 'Close']].values[-1].reshape(1, -1)
    future_predictions = []

    for _ in range(forecast_days):
        future_price = model.predict(last_data)[0]
        future_predictions.append(future_price)
        # Update the 'last_data' with new predictions for iterative forecasting
        last_data = np.array([[future_price, future_price, future_price, future_price]])

    return future_predictions

# Button to get stock price and forecast
if st.button('Get Stock Price'):
    if stock_code:
        try:
            # Download stock data using yfinance
            df = yf.download(stock_code, start=start_date, end=end_date)

            # Check if data is available
            if df.empty:
                st.error(f"No data available for {stock_code} in the selected date range.")
            else:
                # Display stock info and graph as before
                st.header(f"Stock Price for {stock_code}")
                fig = px.line(df, x=df.index, y=['Open', 'Close'], title='Stock Prices')
                st.plotly_chart(fig)

                # Train the SVR model
                model, mse, mae = train_svr_model(df)

                # Forecast future stock prices
                future_prices = predict_future_prices(model, df, forecast_days)

                # Display forecasted prices
                future_dates = pd.date_range(end_date, periods=forecast_days + 1)[1:]
                forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Price': future_prices})
                st.write(f"Forecasted Stock Prices for the next {forecast_days} days:")
                st.write(forecast_df)

                # Plot forecasted prices
                fig_forecast = px.line(forecast_df, x='Date', y='Forecasted Price', title='Forecasted Stock Prices')
                st.plotly_chart(fig_forecast)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a valid stock code.")
