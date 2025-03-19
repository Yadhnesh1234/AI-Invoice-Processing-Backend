from pydantic import BaseModel, Field
from typing import Optional
from typing import List
from datetime import date, datetime

class User(BaseModel):
    user_id: str = Field(..., example="U67890")
    name: str = Field(..., example="John Doe")
    email: Optional[str] = Field(None, example="johndoe@example.com")
    
    business_name: Optional[str] = Field(None, example="John's Electronics")
    business_address: Optional[str] = None

    invoices: List[str] = Field(default=[])  
    products: List[str] = Field(default=[]) 

    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True

