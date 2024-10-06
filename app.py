import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime as dt

# Title of the app
st.title("Welcome to the Stock Dash App!")

# User input for stock code
stock_code = st.text_input("Enter Stock Code")

# Date range picker
start_date = st.date_input("Start Date", value=dt(2021, 1, 1))
end_date = st.date_input("End Date", value=dt.now())

# Input for forecast days
forecast_days = st.number_input("Days of Forecast", min_value=1)

# Button to get stock price
if st.button('Get Stock Price'):
    if stock_code:
        try:
            # Download stock data using yfinance
            ticker = yf.Ticker(stock_code)
            info = ticker.info
            df = yf.download(stock_code, start=start_date, end=end_date)

            # Check if data is available
            if df.empty:
                st.error(f"No data available for {stock_code} in the selected date range.")
            else:
                # Display company info
                logo_url = info.get('logo_url', None)
                if logo_url:
                    st.image(logo_url, width=100)
                else:
                    st.warning("Logo not available.")
                    
                st.header(info.get('shortName', ''))
                st.write(info.get('longBusinessSummary', 'No description available'))

                # Create stock price plot
                fig = px.line(df, x=df.index, y=['Open', 'Close'], title='Stock Prices')
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a valid stock code.")
