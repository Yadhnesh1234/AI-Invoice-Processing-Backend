from fastapi import APIRouter, HTTPException,Depends,Body, File, UploadFile
from services.invoice_data_extract import get_batched_embeddings,semantic_search,gemini_output,generate_invoice_number,llm_recomendation,serialize_objectid,parse_date,get_cleaned_values,get_product_stock
from pymongo.collection import Collection
import json
import re
from db.database import database
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
from bson import ObjectId
from typing import List  
import faiss
import numpy as np

router = APIRouter()
index = []
@router.post("/save-invoice")
async def save_invoice(invoice: UploadFile = File(...)):
        collection: Collection = database["Invoice"]
        file_location = f"./uploaded_invoices/{invoice.filename}"
        with open(file_location, "wb") as file:
              file.write(await invoice.read())
        # temp_file_path = f"./test_img/handwritten_img5.jpg"
        invoice_data = gemini_output(file_location)
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

@router.get("/embed-product/")
async def embedd_products():
    try:
        global index 
        collection: Collection = database['Product']
        products_data = await collection.find({}).to_list(None)
        texts_data = []
        for p in products_data:
            if not p.get("Description"):
               continue
            text = p.get('Description', '')
            texts_data.append(text)
        doc_emb = get_batched_embeddings (texts_data)
        dim = len(doc_emb[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(doc_emb).astype("float32"))
        print(index)
        return {"status":200}
    except Exception as e: 
       raise HTTPException(status_code=500, detail=f"Error While Getting Recommendations: {str(e)}")  
class recommendation(BaseModel):
    search_query:str
    
@router.post("/get-recommendation/")
async def get_recommendation(search:recommendation):
   try:
       global index
       print(index)
       collection: Collection = database['Product']
       products = await collection.find({}).to_list(None)
       prod_desc = [pd['Description'] for pd in products if pd.get('Description')]
       results = semantic_search(prod_desc,index,search.search_query)
    #    match_product = [prod for prod in products if prod['Description']==search.search_query]
    #    if len(match_product)==1:
    #              return {"status":200,"data":match_product[0]}  
    #    print(search.search_query)  
    #    product_names = [product.get('Description', '') for product in products]
    #    product_list=llm_recomendation(search.search_query,product_names)
       desc_to_stock = {prod["Description"]:prod["StockCode"] for prod in products}
       suggest_prod = [{desc_to_stock[desc]:desc} for desc in results if desc in desc_to_stock]
       print(results)
       return {"status":200,"data":suggest_prod}   
   except Exception as e: 
       raise HTTPException(status_code=500, detail=f"Error While Getting Recommendations: {str(e)}")  
             
@router.get("/process-invoice/")
async def process_invoice(id):
    try:
        global index
        collection: Collection = database["Invoice"]
        await collection.update_many(
            {"_id": ObjectId(id)},
            [
                {"$set":{"created_at":{"$toDate":"$created_at"}}}
            ]
        )
        result = await collection.find_one({"_id": ObjectId(id)})
        response = await get_cleaned_values(result,index)
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
    old_descs: List[str]
    new_stock_codes: List[int]

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
        
@router.post("/update-invoice-products/")
async def update_invoice_products(update_data: UpdateInvoiceProduct):
    try:
        collection: Collection = database["Invoice"]
        products: Collection = database["Product"]
        print(update_data)
        if len(update_data.new_stock_codes) != len(update_data.old_descs):
            raise HTTPException(status_code=400, detail="Mismatched lengths of stock codes and descriptions")

        for new_stock_code, old_desc in zip(update_data.new_stock_codes, update_data.old_descs):
            match_prod = await products.find_one({"StockCode": new_stock_code})

            if not match_prod:
                raise HTTPException(status_code=400, detail=f"Invalid stock code: {new_stock_code}")

            stock_code = int(match_prod["StockCode"])
            description = str(match_prod["Description"])
            unit_price = float(match_prod["Price"])

            result = await collection.update_one(
                {
                    "_id": ObjectId(update_data.invoice_id),
                    "ProductItems.Description": old_desc
                },
                {
                    "$set": {
                        "ProductItems.$.StockCode": stock_code,
                        "ProductItems.$.Description": description,
                        "ProductItems.$.UnitPrice": unit_price
                    }
                }
            )

            if result.modified_count == 0:
                raise HTTPException(status_code=404, detail=f"Product with description '{old_desc}' not found in invoice")

        return {"message": "All product descriptions updated successfully"}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating invoice products: {str(e)}")
