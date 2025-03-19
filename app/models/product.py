from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId

class Product(BaseModel):
    UserId: str = Field(..., example="U67890") 
    StockCode: int = Field(..., example="P12345")   
    Description: str = Field(..., example="Laptop")
    Category: Optional[str] = Field(None, example="Electronics")
    Price: float = Field(..., example=50000.00)
    stock: Optional[int] = Field(0, example=10) 

    class Config:
        orm_mode = True