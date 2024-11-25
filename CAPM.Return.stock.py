# Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import datetime
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Streamlit configuration
st.set_page_config(
    page_title="CAPM",
    page_icon="chart_with_upwards_trend",
    layout='wide'
)

st.title("Capital Asset Pricing Model")

# User input for stocks and timeframe
col1, col2 = st.columns([1, 1])
with col1:
    stocks_list = st.multiselect(
        "Choose up to 4 stocks",
        ('TSLA', 'AAPL', 'NFLX', 'MSFT', 'AMZN', 'NVDA', 'GOOGL'),
        ['TSLA', 'AAPL', 'AMZN', 'GOOGL']
    )
with col2:
    year = st.number_input("Number of years", 1, 10)

# Downloading SP500 data
end = datetime.date.today()
start = end - datetime.timedelta(days=year * 365)

try:
    SP500 = web.DataReader(['sp500'], 'fred', start, end)
    SP500.columns = ['sp500']
    SP500.reset_index(inplace=True)
except Exception as e:
    st.error("Failed to fetch SP500 data. Please check your connection or input parameters.")

# Downloading stock data
stocks_df = pd.DataFrame()

for stock in stocks_list:
    try:
        data = yf.download(stock, period=f'{year}y')
        stocks_df[stock] = data['Close']
    except Exception as e:
        st.warning(f"Failed to fetch data for {stock}: {e}")

# Process and merge data
stocks_df.reset_index(inplace=True)
stocks_df['Date'] = pd.to_datetime(stocks_df['Date']).dt.tz_localize(None)
SP500['Date'] = pd.to_datetime(SP500['DATE']).dt.tz_localize(None)

# Merge on Date
stocks_df = pd.merge(stocks_df, SP500, on='Date', how='inner')

# Display head and tail of the data
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### DataFrame Head")
    st.dataframe(stocks_df.head(), use_container_width=True)
with col2:
    st.markdown("### DataFrame Tail")
    st.dataframe(stocks_df.tail(), use_container_width=True)

# Normalize stock prices
def normalize(df):
    normalized_df = df.copy()
    for column in stocks_list:
        normalized_df[column] = df[column] / df[column].iloc[0]
    return normalized_df

# Plotting stock prices
def plot_stock_prices(df, stocks):
    fig = go.Figure()
    for stock in stocks:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[stock], mode='lines', name=stock))
    fig.update_layout(title="Stock Prices Over Time", xaxis_title="Date", yaxis_title="Price (USD)")
    return fig

# Display original and normalized stock prices
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### Price of All Stocks")
    st.plotly_chart(plot_stock_prices(stocks_df, stocks_list))
with col2:
    st.markdown("### Normalized Stock Prices")
    normalized_df = normalize(stocks_df)
    st.plotly_chart(plot_stock_prices(normalized_df, stocks_list))

# Calculate daily returns
def daily_return(df):
    returns_df = df[stocks_list + ['sp500']].pct_change().dropna()
    returns_df['Date'] = df['Date'][1:].values  # Align dates with returns
    return returns_df

stocks_daily_return = daily_return(stocks_df)
st.markdown("### Daily Returns of Selected Stocks")
st.dataframe(stocks_daily_return.head(), use_container_width=True)

# Calculate Beta and Alpha using CAPM
def calculate_beta(df, stock):
    x = df['sp500'].values.reshape(-1, 1)  # Market returns
    y = df[stock].values  # Stock returns
    
    # Remove invalid values
    valid_mask = ~np.isnan(x.flatten()) & ~np.isnan(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    # Perform regression
    model = LinearRegression()
    model.fit(x, y)
    beta = model.coef_[0]
    alpha = model.intercept_
    return beta, alpha

beta = {}
alpha = {}

for stock in stocks_list:
    try:
        b, a = calculate_beta(stocks_daily_return, stock)
        beta[stock] = b
        alpha[stock] = a
    except Exception as e:
        st.warning(f"Could not calculate Beta/Alpha for {stock}: {e}")

# Display calculated Beta values
beta_df = pd.DataFrame({'Stock': beta.keys(), 'Beta Value': [round(b, 2) for b in beta.values()]})
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('### Calculated Beta Values')
    st.dataframe(beta_df, use_container_width=True)

# Calculate expected returns using CAPM
rf = 0.03  # Example risk-free rate (3%)
rm = stocks_daily_return['sp500'].mean() * 252  # Annualized market return

return_df = pd.DataFrame({
    'Stock': stocks_list,
    'Expected Return': [round(rf + beta[stock] * (rm - rf), 2) for stock in stocks_list]
})

with col2:
    st.markdown('### Expected Returns (CAPM)')
    st.dataframe(return_df, use_container_width=True)
