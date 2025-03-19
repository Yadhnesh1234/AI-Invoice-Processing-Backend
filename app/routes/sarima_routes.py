from fastapi import APIRouter
from services.sarima_service import data_process, get_top_products, prepare_time_series,get_forecast
router = APIRouter()


@router.get("/get-price-forecast-product")
async def get_forecast_for_product(stockcode: int = None):
    df = data_process()
    
    if stockcode:
        stock_codes = [stockcode]
    else:
        top_products = get_top_products(df, n=5)
        stock_codes = top_products.index.tolist()

    response = {}
    for stock in stock_codes:
        weekly_ts = prepare_time_series(df, stock, freq="W")
        weekly_forecast = get_forecast(weekly_ts, freq="W")
        
        monthly_ts = prepare_time_series(df, stock, freq="M")
        monthly_forecast = get_forecast(monthly_ts, freq="M")
        
        response[stock] = {
            "weekly": {
                "dates": weekly_forecast.index.strftime("%Y-%m-%d").tolist(),
                "historical": weekly_forecast["total_price"].dropna().tolist(),
                "model1": weekly_forecast["model1"].tolist(),
                "model2": weekly_forecast["model2"].tolist()
            },
            "monthly": {
                "dates": monthly_forecast.index.strftime("%Y-%m-%d").tolist(),
                "historical": monthly_forecast["total_price"].dropna().tolist(),
                "model1": monthly_forecast["model1"].tolist(),
                "model2": monthly_forecast["model2"].tolist()
            }
        }
    
    return response