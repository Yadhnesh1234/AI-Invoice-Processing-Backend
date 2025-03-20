from db.data import ds
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def get_top_products(df, n=5, by=None):
    return df.groupby("StockCode")[by].sum().sort_values(ascending=False)[:n]

def data_process():
    df = ds
    df['total_price'] = df['Price'] * df['Quantity']
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"]).set_index("InvoiceDate")  
    df = df.sort_index()
    return df

def sarima_fun(test_ts, freq="W", periods=20):
    df2 = ds
    df2["InvoiceDate"] = pd.to_datetime(df2["InvoiceDate"], errors="coerce")
    df2['total_price'] = df2['Price'] * df2['Quantity']
    test_ts = test_ts.dropna()
    ts2 = df2.set_index("InvoiceDate")["total_price"]
    ts2 = np.log(ts2.sort_index())
    ts2[ts2 == np.inf] = 0
    ts2 = pd.DataFrame(ts2.resample(freq).mean())  # Resample based on freq
    df2 = pd.DataFrame(test_ts.fillna(0))
    new_df = pd.concat([df2, ts2.fillna(1)])
   
    # Adjust seasonal order based on frequency (daily: 7, weekly: 52, monthly: 12)
    seasonal_period = {"D": 7, "W": 52, "M": 12}.get(freq, 52)
    model1 = SARIMAX(test_ts, order=(2,1,0), trend='n', time_varying_regression=True,
                     mle_regression=False, seasonal_order=(1,1,1,seasonal_period)).fit()
    model2 = SARIMAX(test_ts, order=(0,1,2), trend='n', time_varying_regression=True,
                     mle_regression=False, seasonal_order=(1,1,1,seasonal_period)).fit()

    # Generate future dates based on frequency
    if freq == "D":
        future_dates = pd.date_range(start=test_ts.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    elif freq == "W":
        future_dates = pd.date_range(start=test_ts.index[-1] + pd.Timedelta(weeks=1), periods=periods, freq="W")
    elif freq == "M":
        future_dates = pd.date_range(start=test_ts.index[-1] + pd.offsets.MonthEnd(1), periods=periods, freq="M")

    forecast_df = pd.DataFrame(index=future_dates)
    forecast_df["model1"] = model1.forecast(steps=periods).values
    forecast_df["model2"] = model2.forecast(steps=periods).values
    new_df = pd.concat([test_ts.to_frame(name="total_price"), forecast_df])
    new_df = np.exp(new_df)
    y_true = new_df[~new_df["model1"].isnull()]["total_price"]
    y_pred_1 = new_df[~new_df["model1"].isnull()]["model1"]
    y_pred_2 = new_df[~new_df["model1"].isnull()]["model2"]
    print(f"Predictions for Model 1 ({freq}):\n{y_pred_1}")
    print(f"Predictions for Model 2 ({freq}):\n{y_pred_2}")
    return new_df  

def get_forecast_preprocess(stock_code=None, freq="W", periods=20):
    df = data_process()
   
    if stock_code == 0:
        top_p_r = get_top_products(df, 5, 'total_price')
        print("data : ",top_p_r)
        stock_codes = top_p_r.index[:5]  
    else:
        stock_codes = [stock_code]

    ts = df[["Quantity", "StockCode", "total_price"]]
    units_ts = ts[["total_price", "StockCode"]]
    test_ts = None
    stock_code_descriptions = {}
    for prod_id in stock_codes:  
        product = prod_id
        product_description = df[df["StockCode"] == prod_id]["Description"].iloc[0] if "Description" in df.columns else "No description available"
        stock_code_descriptions[prod_id] = product_description
        print(stock_code_descriptions)
        d_range_s = "2009-12"
        d_range_e = "2011-12"
        new_ts = units_ts[units_ts["StockCode"] == prod_id]["total_price"]
        new_ts2 = new_ts[d_range_s:d_range_e].resample(freq).max()  
        new_ts3 = new_ts[d_range_s:d_range_e].resample(freq).mean()  
        test_ts = np.log(new_ts3)
   
    result = sarima_fun(test_ts, freq=freq, periods=periods)
    print(f"Forecast Results ({freq}):\n{result}")
    result_dict = {
        "frequency": freq,
        "periods": periods,
        "stock_codes": [
            {
                "stock_code": code,
                "description": stock_code_descriptions.get(code, "No description available")  # Add description to result
            }
            for code in stock_codes
        ],

        "forecast": [
            {
                "date": index.strftime("%Y-%m-%d"),  
                "total_price": None if pd.isna(row["total_price"]) else round(row["total_price"], 2),
                "model1": None if pd.isna(row["model1"]) else round(row["model1"], 2),
                "model2": None if pd.isna(row["model2"]) else round(row["model2"], 2)
            }
            for index, row in result.iterrows()
        ]
    }
    return result_dict


def sarima_quantity_fun(test_ts, freq="W", periods=20):
    df2 = ds
    df2["InvoiceDate"] = pd.to_datetime(df2["InvoiceDate"], errors="coerce")
    df2['total_price'] = df2['Price'] * df2['Quantity']
    test_ts = test_ts.dropna()
    ts2 = df2.set_index("InvoiceDate")["Quantity"]
    ts2 = np.log(ts2.sort_index())
    ts2[ts2 == np.inf] = 0
    ts2 = pd.DataFrame(ts2.resample(freq).mean())  # Resample based on freq
    df2 = pd.DataFrame(test_ts.fillna(0))
    new_df = pd.concat([df2, ts2.fillna(1)])
   
    # Adjust seasonal order based on frequency (daily: 7, weekly: 52, monthly: 12)
    seasonal_period = {"D": 7, "W": 52, "M": 12}.get(freq, 52)
    model1 = SARIMAX(test_ts, order=(2,1,0), trend='n', time_varying_regression=True,
                     mle_regression=False, seasonal_order=(1,1,1,seasonal_period)).fit()
    model2 = SARIMAX(test_ts, order=(0,1,2), trend='n', time_varying_regression=True,
                     mle_regression=False, seasonal_order=(1,1,1,seasonal_period)).fit()

    # Generate future dates based on frequency
    if freq == "D":
        future_dates = pd.date_range(start=test_ts.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    elif freq == "W":
        future_dates = pd.date_range(start=test_ts.index[-1] + pd.Timedelta(weeks=1), periods=periods, freq="W")
    elif freq == "M":
        future_dates = pd.date_range(start=test_ts.index[-1] + pd.offsets.MonthEnd(1), periods=periods, freq="M")

    forecast_df = pd.DataFrame(index=future_dates)
    forecast_df["model1"] = model1.forecast(steps=periods).values
    forecast_df["model2"] = model2.forecast(steps=periods).values
    new_df = pd.concat([test_ts.to_frame(name="Quantity"), forecast_df])
    new_df = np.exp(new_df)
    y_true = new_df[~new_df["model1"].isnull()]["Quantity"]
    y_pred_1 = new_df[~new_df["model1"].isnull()]["model1"]
    y_pred_2 = new_df[~new_df["model1"].isnull()]["model2"]
    print(f"Predictions for Model 1 ({freq}):\n{y_pred_1}")
    print(f"Predictions for Model 2 ({freq}):\n{y_pred_2}")
    return new_df  

def get_quantity_forecast_preprocess(stock_code=None, freq="W", periods=20):
    df = data_process()
   
    if stock_code == 0:
        top_p_r = get_top_products(df, 5, 'Quantity')
        print("data : ",top_p_r)
        stock_codes = top_p_r.index[:5]  
    else:
        stock_codes = [stock_code]

    ts = df[["Quantity", "StockCode", "total_price"]]
    units_ts = ts[["Quantity", "StockCode"]]
    test_ts = None
    stock_code_descriptions = {}
    for prod_id in stock_codes:  
        product = prod_id
        product_description = df[df["StockCode"] == prod_id]["Description"].iloc[0] if "Description" in df.columns else "No description available"
        stock_code_descriptions[prod_id] = product_description
        print(stock_code_descriptions)
        d_range_s = "2009-12"
        d_range_e = "2011-12"
        new_ts = units_ts[units_ts["StockCode"] == prod_id]["Quantity"]
        new_ts2 = new_ts[d_range_s:d_range_e].resample(freq).max()  
        new_ts3 = new_ts[d_range_s:d_range_e].resample(freq).mean()  
        test_ts = np.log(new_ts3)
   
    result = sarima_quantity_fun(test_ts, freq=freq, periods=periods)
    print(f"Forecast Results ({freq}):\n{result}")
    result_dict = {
        "frequency": freq,
        "periods": periods,
        "stock_codes": [
            {
                "stock_code": code,
                "description": stock_code_descriptions.get(code, "No description available") 
            }
            for code in stock_codes
        ],

        "forecast": [
            {
                "date": index.strftime("%Y-%m-%d"),  
                "quantity": None if pd.isna(row["Quantity"]) else round(row["Quantity"], 2),
                "model1": None if pd.isna(row["model1"]) else round(row["model1"], 2),
                "model2": None if pd.isna(row["model2"]) else round(row["model2"], 2)
            }
            for index, row in result.iterrows()
        ]
    }
    return result_dict