from fastapi import APIRouter, HTTPException,Depends,Body
from services.invoice_data_extract import gemini_output,generate_invoice_number,llm_recomendation,serialize_objectid,parse_date,get_cleaned_values,get_product_stock
from pymongo.collection import Collection
import json
import re
from db.database import database
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
from bson import ObjectId

router = APIRouter()

@router.get("/save-invoice")
async def save_invoice():
        collection: Collection = database["Invoice"]
        temp_file_path = f"./test_img/handwritten_img5.jpg"
        invoice_data = gemini_output(temp_file_path)
        json_string = re.sub(r'```json\n(.*?)\n```', r'\1', invoice_data, flags=re.DOTALL)
        parsed_data = json.loads(json_string)
        parsed_data['status']=False
        invoice_number = parsed_data.get("InvoiceNo")
        if invoice_number is None:
          now = datetime.now()
          current_year = now.year
          current_month = now.month
          counter_invoice=await generate_invoice_number()
          invoice_number = str(current_year)+str(current_month)+str(counter_invoice)
        parsed_data['InvoiceNo']=invoice_number
        result=await collection.insert_one(parsed_data)
        return {"status":200,"data":str(result.inserted_id)}

@router.get("/get-invoice/")
async def get_invoice(id: str):
    try:
        collection: Collection = database['Invoice']
        data = await collection.find_one({"_id": ObjectId(id)})
        
        if data is None:
            raise HTTPException(status_code=404, detail="Invoice not found")
        data["_id"] = str(data["_id"])
        return {"response": data}
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid invoice ID format: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing invoice: {str(e)}")

class recommendation(BaseModel):
    search_query:str
    
@router.post("/get-recommendation/")
async def get_recommendation(search:recommendation):
   try:
       collection: Collection = database['Product']
       products = await collection.find({}).to_list(None)
       match_product = [prod for prod in products if prod['Description']==search.search_query]
       if len(match_product)==1:
                 return {"status":200,"data":match_product[0]}  
       print(search.search_query)  
       product_names = [product.get('Description', '') for product in products]
       product_list=llm_recomendation(search.search_query,product_names)
       desc_to_stock = {prod["Description"]:prod["StockCode"] for prod in products}
       suggest_prod = [{desc_to_stock[desc]:desc} for desc in product_list if desc in desc_to_stock]
       return {"status":200,"data":suggest_prod}   
   except Exception as e: 
       raise HTTPException(status_code=500, detail=f"Error While Getting Recommendations: {str(e)}")  
             
@router.get("/process-invoice/")
async def process_invoice(id):
    try:
        collection: Collection = database["Invoice"]
        await collection.update_many(
            {"_id": ObjectId(id)},
            [
                {"$set":{"created_at":{"$toDate":"$created_at"}}}
            ]
        )
        result = await collection.find_one({"_id": ObjectId(id)})
        response = await get_cleaned_values(result)
        if(response["code"]==0):
            return {"Status":201,"Response": "Ambiguous Products","data":response["data"]}
        if(response["code"]==-1):
            return {"Status":202,"Response": "Product Not Present","data":response["data"]}
        result = await collection.update_one({"_id": ObjectId(id)},{"$set":response["data"]})
        return {"Status":200,"Response": "Data Is Right","data":response["data"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing invoice: {str(e)}")



class UpdateInvoiceProduct(BaseModel):
    invoice_id: str
    old_desc: str
    new_stock_code:int

class UpdateInvoiceDate(BaseModel):
    invoice_id: str
    date:datetime
    
@router.post("/update-invoice-date")
async def update_invoice_date(update_data: UpdateInvoiceDate):
    try:
        collection: Collection = database["Invoice"]
        result = await collection.update_one(
            {
                "_id": ObjectId(update_data.invoice_id)
            },
            {
                "$set": {
                    "InvoiceDate":update_data.date
                }
            }
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Invoice not found")

        return {"message": "Invoice Date updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating invoice Date: {str(e)}")
        
@router.post("/update-invoice-product/")
async def update_invoice_product(update_data: UpdateInvoiceProduct):
    try:
        collection: Collection = database["Invoice"]
        products: Collection = database["Product"]

        print(update_data.new_stock_code)  

        match_prod = await products.find_one({"StockCode": update_data.new_stock_code})

        if not match_prod:
            raise HTTPException(status_code=400, detail="Invalid stock code")

        print("Match Product:", match_prod)

        # Ensure correct data types
        stock_code = int(match_prod["StockCode"])
        description = str(match_prod["Description"])
        unit_price = float(match_prod["Price"])  

        result = await collection.update_one(
            {
                "_id": ObjectId(update_data.invoice_id),
                "ProductItems.Description": update_data.old_desc
            },
            {
                "$set": {
                    "ProductItems.$.StockCode": stock_code,
                    "ProductItems.$.Description": description,
                    "ProductItems.$.UnitPrice": unit_price,  
                }
            }
        )

        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Invoice not found")

        return {"message": "Product description updated successfully"}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating invoice product: {str(e)}")
