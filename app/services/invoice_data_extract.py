import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import os
from fastapi import HTTPException,Depends
from db.database import database
from datetime import datetime
from pymongo.collection import Collection
from bson import ObjectId
from transformers import AutoTokenizer
from motor.motor_asyncio import AsyncIOMotorClient
import cohere
import numpy as np


load_dotenv()
co = cohere.ClientV2("UZHbYbgFigQvfkHwNwNs2564Yx4T8yWVB6jHsFhs")

DATE_FORMATS = [
    "%d/%m/%Y", 
    "%m/%d/%Y",  
    "%Y-%m-%d",  
    "%d-%m-%Y", 
    "%b %d, %Y", 
]


def image_format(image_path):
    img = Path(image_path)

    if not img.exists():
        raise FileNotFoundError(f"Could not find image: {img}")

    image_parts = [
        {"mime_type": "image/jpeg", "data": img.read_bytes()}  
    ]
    return image_parts

def gemini_output(image_path):
    gene_ai_key = os.getenv('GENAI_API_KEY')
    print(gene_ai_key)
    genai.configure(api_key=gene_ai_key)

    MODEL_CONFIG = {
         "temperature": 0.2,
         "top_p": 1,
         "top_k": 32,
         "max_output_tokens": 4096,
    }

    safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    ]

    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=MODEL_CONFIG, safety_settings=safety_settings)
    
    try:
        system_prompt = """
               You are a specialist in comprehending receipts.
               Input images in the form of receipts will be provided to you,
               and your task is to respond to questions based on the content of the input image.
               """               
        user_prompt = """
                Please extract the data from the invoice image and convert it into a JSON format. assign  proper values to  following fields that are included in the JSON structure, if some fields need calculation then make proper calculations and if any field is missing in the invoice, assign it as `null` or an empty string (`""`). The fields are:
                {
                    "InvoiceNo": null,
                    "InvoiceDate": null,
                    "SellerName": null,
                    "SellerAddress": null,
                    "Customer ID":null,
                    "Customer Name": null,
                    "ProductItems": [
                        {
                        "Description": null,
                        "StockCode": null,
                        "Category": null,
                        "Quantity": null,
                        "UnitPrice":null,
                        "total_price": null
                        }
                    ],
                    "SubTotal": null,
                    "TotalAmount": null,
                    "created_at": null,
                    "updated_at": null
                    }
                Make sure the values are extracted accurately from the invoice. If any of these fields are not present in the invoice, please assign `null` (or an empty string for strings) as the value for that field. Thank you!
                """
        image_info = image_format(image_path)
        input_prompt = [system_prompt, image_info[0], user_prompt]
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting data from invoice: {str(e)}")

async def generate_invoice_number():
     collection: Collection = database["Invoice"]
     pipeline = [
        {
            "$project": {
                "year": {"$year": "$created_at"},  
                "month": {"$month": "$created_at"}
            }
        },
        {
            "$match": {
                "year": datetime.now().year,  
                "month": datetime.now().month  
            }
        }
      ]
     result = await collection.aggregate(pipeline).to_list(length=None)
     if not result:
        return 1
     last_counter = len(result)
    
     next_counter = last_counter + 1
    
     return  next_counter

def parse_date(date_str: str) -> str:
    for date_format in DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(date_str, date_format)
            return parsed_date.strftime("%Y-%m-%d")  
        except ValueError:
            continue  
    return None     
 
def serialize_objectid(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: serialize_objectid(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_objectid(item) for item in obj]
    return obj

def semantic_search(products,index,query):
    query_emb = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    ).embeddings.float
    D, I = index.search(np.array(query_emb).astype("float32"), k=5)
    results = [products[i] for i in I[0]]
    return results
def llm_recomendation(search_query,product_names):
     gene_ai_key = os.getenv('GENAI_API_KEY')
     genai.configure(api_key=gene_ai_key)
     MODEL_CONFIG = {
         "temperature": 0.2,
         "top_p": 1,
         "top_k": 32,
         "max_output_tokens": 4096,
       }

     safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
       ]
     model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=MODEL_CONFIG, safety_settings=safety_settings)
     system_prompt = """"You are an advanced search assistant. You have provided a list of product descriptions and search query. you have to check for  matches or identical products or semantic match  results from the **Product description list only**."""
     input_prompt = f"{system_prompt}\n\n"+"Product Descriptions List:\n" + "\n".join(
        [f"{desc}" for desc in enumerate(product_names)]
    ) +"Search Query:"+ search_query+ "\n\nInstructions: **only return product desriptions as it is from Product Description list if product name contain any symblos also include that also** not there serial numbers no special symbols"
     response = model.generate_content(input_prompt)   
     print("Response text : ",response.text)   
     product_list=response.text.split('\n')
     product_list=list(filter(lambda x:x!='',product_list))   
     return product_list 
async def get_product_stock(data,index):
  try:
       collection: Collection = database['Product']
       products = await collection.find({}).to_list(None)
       search_query =data["Description"]
       match_product = [prod for prod in products if prod['Description']==search_query]
       if len(match_product)==1:
               return {"code":1,"data":match_product[0]} 
       product_names = [pd['Description'] for pd in products if pd.get('Description')]
       product_list=semantic_search(product_names,index,search_query)#llm_recomendation(search_query,product_names)
       print("Product List: ",product_list)
       if len(product_list)>1: 
           desc_to_stock = {prod["Description"]:prod["StockCode"] for prod in products}
           suggest_prod = [{desc_to_stock[desc]:desc} for desc in product_list if desc in desc_to_stock]
           print(suggest_prod)
           return {"code":0,"data":suggest_prod}
       match_product=[]
       match_product = [prod for prod in products if prod['Description']==product_list[0]]
       if(len(match_product)==0):
         return {"code":-1,"data":-1}
       return {"code":1,"data":match_product[0]}
  except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting data from invoice: {str(e)}")
    
    
    
async def get_cleaned_values(parsed_data,index):           
     invoice_date = parsed_data.get("InvoiceDate")
     if isinstance(invoice_date, str):
         invoice_date = parse_date(invoice_date)  
     elif isinstance(invoice_date, datetime):
         invoice_date = invoice_date.strftime("%Y-%m-%d %H:%M:%S")
     else:
         invoice_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
     total_amount = parsed_data.get("TotalAmount")
     if not total_amount:
            product_items = parsed_data.get("ProductItems", [])
            if(len(product_items)==0):
                return {"code":-1,"data":-1}
            total_amount = sum(item.get("total_price", 0) for item in product_items)
     flag=0
     suggetion_list=[]
     print("Hello")
     for data in parsed_data.get("ProductItems"):
           print("DATA BEFORE get_product_stock:", data)
           response=await get_product_stock(data,index)
           if(response["code"]==0):
               flag=1
               suggetion_list.append({"Description":data["Description"],"Items":response["data"]})
           elif response["code"] == -1:   
                return {"code":-1,"data":data["Description"]}
           else :
                if data["StockCode"] is None:
                      result = response["data"]
                      data["Description"] = result["Description"]
                      data["StockCode"] = result["StockCode"]
                      data["UnitPrice"] = result["Price"]
                      data["Quantity"] = float(data["total_price"]) / float(result["Price"])
                elif response["code"] == 1:
                      result = response["data"]
                      data["Quantity"] = float(data["total_price"]) / float(result["Price"])
               
     if flag :
         return {"code":0,"data":suggetion_list}    
     subtotal = parsed_data.get("SubTotal")
     if not subtotal:
            product_items = parsed_data.get("ProductItems", [])
            subtotal = sum(float(item.get("total_price", 0)) for item in product_items)
     seller_name = parsed_data.get("SellerName", "")
     seller_address = parsed_data.get("SellerAddress", "")
     invoice = {
            "InvoiceNo": parsed_data.get("InvoiceNo",""),
            "InvoiceDate": invoice_date,
            "SellerName": seller_name,
            "SellerAddress": seller_address,
            "Customer ID": parsed_data.get("Customer ID", ""),
            "Customer Name": parsed_data.get("Customer Name", ""),
            "ProductItems": parsed_data.get("ProductItems", []),
            "SubTotal": subtotal,
            "TotalAmount": total_amount,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "status":True
        }
     return {"code":1,"data":invoice}

def get_batched_embeddings(texts, batch_size=96):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = co.embed(
            texts=batch,
            model="embed-multilingual-v3.0",
            input_type="search_document",
            embedding_types=["float"]
        ).embeddings.float
        all_embeddings.extend(response)
    return all_embeddings