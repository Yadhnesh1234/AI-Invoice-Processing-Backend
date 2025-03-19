from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, date

class InvoiceProduct(BaseModel):
    Description: str = Field(..., example="Sugar")
    StockCode: Optional[str] = Field(None, example="P12345")
    Category: Optional[str] = Field(None, example="Groceries")
    Quantity: int = Field(..., example=1)
    UnitPrice: float = Field(..., example=240.00)
    total_price: float = Field(..., example=240.00)
    class Config:
        orm_mode = True

class Invoice(BaseModel):
    Invoice: Optional[str] = Field(None, example="INV-1001")
    InvoiceDate: Optional[date] = Field(None, example="2024-02-02")

    SellerName: str = Field(..., example="Guru Krupa Trades")
    SellerAddress: Optional[str] = Field(None, example="meri-link road 18 Sector Nothik")

    CustomerID: Optional[str] = Field(None, example="John Doe")
    CustomerName: Optional[str] = Field(None, example="123 Main Street, City")
    ProductItems: List[InvoiceProduct]  
    SubTotal: Optional[float] = Field(None, example=1580.00)
    TotalAmount: float = Field(..., example=1580.00)
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    status:bool

    class Config:
        orm_mode = True

  