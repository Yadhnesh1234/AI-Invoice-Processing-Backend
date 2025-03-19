from db.data import ds
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def get_top_products(df, n=5, by="total_price"):
    return df.groupby("StockCode")[by].sum().sort_values(ascending=False)[:n]

def data_process():
    df = ds.copy()
    df['total_price'] = df['Price'] * df['Quantity']
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"]).set_index("InvoiceDate")
    df = df.sort_index()
    return df

def prepare_time_series(df, stock_code, freq="W", start_date="2009-12", end_date="2011-12"):
    ts = df[df["StockCode"] == stock_code]["total_price"]
    ts = ts[start_date:end_date].resample(freq).mean()
    return np.log(ts.replace([np.inf, -np.inf], np.nan).dropna())

def fit_sarima_model(ts, order, seasonal_order, steps, freq):
    model = SARIMAX(ts, order=order, trend='n', time_varying_regression=True, 
                    mle_regression=False, seasonal_order=seasonal_order).fit(disp=False)
    last_date = ts.index[-1]  # This is a Timestamp
    if freq == "W":
        start_date = last_date + pd.offsets.Week(1)  # Adds 1 week
    elif freq == "M":
        start_date = (last_date + pd.offsets.MonthEnd(0)).shift(months=1)
    
    future_dates = pd.date_range(start=start_date, periods=steps, freq=freq)
    forecast = model.forecast(steps=steps)
    return pd.Series(np.exp(forecast), index=future_dates)

def get_forecast(ts, freq="W"):
    steps = 20 if freq == "W" else 12  
    seasonal_period = 11 if freq == "W" else 12 
    
    forecast1 = fit_sarima_model(ts, order=(2, 1, 0), seasonal_order=(1, 1, 1, seasonal_period), steps=steps, freq=freq)
    
    forecast2 = fit_sarima_model(ts, order=(0, 1, 2), seasonal_order=(1, 1, 1, seasonal_period), steps=steps, freq=freq)
    
    historical = pd.Series(np.exp(ts), index=ts.index)
    result = pd.concat([historical.rename("total_price"), forecast1.rename("model1"), forecast2.rename("model2")], axis=1)
    return result