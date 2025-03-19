from fastapi import APIRouter, File, UploadFile, HTTPException
from services.prediction_service import combine_csv_files,perform_kmeans_clustering,get_high_recency_prod,get_high_frequency_prod,get_high_monetary_prod,get_high_recency_high_frequency_prod,get_high_amount_high_frequency_prod,get_low_frequency_low_monetary_prod,get_low_recency_low_frequency_prod,get_high_loyalty_products,get_price_sensitive_products,get_potential_high_value_products
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import re
from db.data import ds

router = APIRouter()

@router.get("/frequent-purchase-items")
async def frequent_purchase_items(type: str):
    try:
        rfm_analysis = perform_kmeans_clustering(ds)
        print(ds.columns)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        function_map = {
            "high_recency": get_high_recency_prod,
            "high_frequency": get_high_frequency_prod,
            "high_monetary": get_high_monetary_prod,
            "high_recency_high_frequency": get_high_recency_high_frequency_prod,
            "high_amount_high_frequency": get_high_amount_high_frequency_prod,
            "low_frequency_low_monetary": get_low_frequency_low_monetary_prod,
            "low_recency_low_frequency": get_low_recency_low_frequency_prod,
            "high_loyalty": get_high_loyalty_products,
        }
        if type not in function_map:
            raise HTTPException(status_code=400, detail="Invalid type provided")
        
        data = function_map[type](ds, rfm_analysis)
        
        return {"response": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing invoice: {str(e)}")
    
def safe_eval(value):
    try:
        match = re.match(r"frozenset\((\{.*\})\)", value)
        if match:
            return eval(match.group(1)) 
        return set()  
    except Exception:
        return set()


@router.get("/get-association-rule/")
async def get_association_rule():
    try:
        rules = pd.read_csv("./data/optimized_association_rules_new.csv")
        rules["antecedents"] = rules["antecedents"].apply(safe_eval)
        rules["consequents"] = rules["consequents"].apply(safe_eval)

        # Ensure rules with valid sets are selected
        rules = rules[(rules["antecedents"].apply(len) > 0) & (rules["consequents"].apply(len) > 0)]

        # Convert sets to lists for JSON serialization
        rules_json = rules.copy()
        rules_json["antecedents"] = rules_json["antecedents"].apply(list)
        rules_json["consequents"] = rules_json["consequents"].apply(list)

        return {"association_rules": rules_json[["antecedents", "consequents", "confidence", "lift"]].to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing invoice: {str(e)}")