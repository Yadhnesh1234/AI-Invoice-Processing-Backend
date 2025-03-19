from fastapi import APIRouter, HTTPException
from models.product import Product
from db.database import database
from bson import ObjectId

router = APIRouter()

@router.post("/products/", response_model=Product)
async def create_product(product: Product):
    product_dict = product.dict()
    result = await database["products"].insert_one(product_dict)
    product_dict["_id"] = str(result.inserted_id)
    return product_dict

@router.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: str):
    product = await database["products"].find_one({"_id": ObjectId(product_id)})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    product["_id"] = str(product["_id"])
    return product
