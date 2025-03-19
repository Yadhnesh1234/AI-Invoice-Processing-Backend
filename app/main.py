from fastapi import FastAPI
from routes.user_routes import router as user_router
from routes.product_routes import router as product_router
from routes.sarima_routes import router as forecast_router
from routes.invoice_routes import router as invoice_router
from routes.predictions_routes import router as prediction_router
from pymongo.collection import Collection
from db.database import database
import pandas as pd


app = FastAPI()

app.include_router(user_router, prefix="/api", tags=["Users"])
app.include_router(product_router, prefix="/api", tags=["Products"])
app.include_router(invoice_router, prefix="/api", tags=["Invoices"])
app.include_router(prediction_router, prefix="/api", tags=["Predictions"])
app.include_router(forecast_router, prefix="/api", tags=["forecast"])

@app.get("/")
async def root():
    try:
        await database.command("ping")
        print("MongoDB Connected")
        # collection :Collection= database["Product"]
        # df = pd.read_csv('./data/combine_dataset_2009_2011.csv')
        # df = df[~df["Invoice"].astype(str).str.startswith("C")]
        # df=df.groupby('StockCode').agg({
        #     'Description':'first',
        #     'Price' :'first'
        # }).reset_index()
        # df['Description']=df['Description'].str.strip()
        # data = df.to_dict(orient="records")
        # for entry in data:
        #        entry["UserId"] = "U001"
        # if data:
        #     collection.insert_many(data)
        #     return {"message": "Data uploaded successfully", "inserted_count": len(data)}
        # else:
        #     return {"message": "No valid records to insert"}
    except Exception as e:
        return {"error": "Failed to connect to MongoDB Atlas", "details": str(e)}