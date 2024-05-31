import streamlit as st
from datetime import date
import yfinance as yf
from statsmodels.tsa.ar_model import AutoReg
from plotly import graph_objs as go
import pandas as pd
import numpy as np

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 4  # 4 quarters in a year

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Train the autoregressive model
model = AutoReg(df_train['y'], lags=1)
model_fit = model.fit()

# Make predictions
future_quarters = pd.date_range(start=TODAY, periods=period, freq='Q')
predictions = model_fit.predict(start=len(df_train), end=len(df_train)+len(future_quarters)-1)

# Create a DataFrame for the predictions
df_predictions = pd.DataFrame({
    'Quarter': future_quarters,
    'Predicted Close': predictions
})

st.subheader('Forecast data')
st.table(df_predictions)  # Display the DataFrame as a table

st.write(f'Forecast plot for {n_years} years')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=future_quarters, y=predictions, name="Predicted Close"))
fig1.layout.update(title_text='Stock Price Predictions', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)
