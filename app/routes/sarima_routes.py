from fastapi import APIRouter
from services.sarima_service import get_forecast_preprocess
router = APIRouter()


@router.get("/get-forecast-product")
async def get_forecast_for_product(stockcode:int):
    response=get_forecast_preprocess(stockcode, freq="M", periods=12)
    return response