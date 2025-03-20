from fastapi import APIRouter
from services.sarima_service import get_forecast_preprocess,get_quantity_forecast_preprocess
from pydantic import BaseModel

router = APIRouter()

class freqmodel(BaseModel):
    stockcode:int
    freq:str
    period:int

@router.post("/get-forecast-product/")
async def get_forecast_for_product(data:freqmodel):
    response=get_forecast_preprocess(data.stockcode, data.freq, data.period)
    return response

class freqmodel(BaseModel):
    stockcode:int
    freq:str
    period:int

@router.post("/get-quantity-forecast-product/")
async def get_quantity_forecast_for_product(data:freqmodel):
    response=get_quantity_forecast_preprocess(data.stockcode, data.freq, data.period)
    return response