import pandas as pd
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def combine_csv_files(file1_path: str, file2_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df = combined_df[combined_df['StockCode'].str.isnumeric()]
    if output_path:
        combined_df.to_csv(output_path, index=False)

    return combined_df

def perform_kmeans_clustering(ds: pd.DataFrame) -> pd.DataFrame:
    ds = ds.dropna(subset=['InvoiceDate', 'StockCode', 'Quantity', 'Price'])
    ds['InvoiceDate'] = pd.to_datetime(ds['InvoiceDate'], errors='coerce')
    
    max_date = ds['InvoiceDate'].max()

    # RFM Calculation
    recency_df = ds.groupby('StockCode')['InvoiceDate'].max().reset_index()
    recency_df['Recency'] = (max_date - recency_df['InvoiceDate']).dt.days
    recency_df.drop(columns=['InvoiceDate'], inplace=True)

    frequency_df = ds.groupby('StockCode')['Invoice'].nunique().reset_index()
    frequency_df.columns = ['StockCode', 'Frequency']

    ds['TotalSales'] = ds['Quantity'] * ds['Price']
    monetary_df = ds.groupby('StockCode')['TotalSales'].sum().reset_index()
    monetary_df.columns = ['StockCode', 'Monetary']

    # Merge RFM Data
    rfm_df = recency_df.merge(frequency_df, on='StockCode').merge(monetary_df, on='StockCode')

    # Handle outliers using IQR
    for col in ['Recency', 'Frequency', 'Monetary']:
        Q1 = rfm_df[col].quantile(0.05)
        Q3 = rfm_df[col].quantile(0.95)
        IQR = Q3 - Q1
        rfm_df = rfm_df[(rfm_df[col] >= Q1 - 1.5 * IQR) & (rfm_df[col] <= Q3 + 1.5 * IQR)]

    # Normalize data
    scaler = MinMaxScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    # Determine optimal K using silhouette score
    best_k, best_score = 3, -1  # Default to 3 clusters
    for k in range(3, 7):
        kmeans = KMeans(n_clusters=k, max_iter=50, random_state=42, n_init=10)
        labels = kmeans.fit_predict(rfm_scaled)
        score = silhouette_score(rfm_scaled, labels)
        if score > best_score:
            best_k, best_score = k, score

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=best_k, max_iter=50, random_state=42, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # TF-IDF for product categorization
    product_descriptions = ds[['StockCode', 'Description']].drop_duplicates().dropna()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    description_vectors = vectorizer.fit_transform(product_descriptions['Description'])

    svd = TruncatedSVD(n_components=5, random_state=42)
    reduced_vectors = svd.fit_transform(description_vectors)

    # KMeans for product categories
    kmeans_desc = KMeans(n_clusters=5, random_state=42, n_init=10)
    category_labels = kmeans_desc.fit_predict(reduced_vectors)

    product_categories = pd.DataFrame({
        'StockCode': product_descriptions['StockCode'].values,
        'ProductCategory': category_labels
    })

    # Merge Product Categories
    rfm_df = rfm_df.merge(product_categories, on='StockCode', how='left')

    return rfm_df

#Identifies products from the cluster with the highest recency, meaning they were purchased recently. Useful for recommending trending or newly popular products.
def get_high_recency_prod(ds: pd.DataFrame, rfm_df: pd.DataFrame) -> list:
    rfm_df = rfm_df.replace([np.inf, -np.inf], np.nan).dropna()
    cluster_summary = rfm_df.groupby('Cluster')['Recency'].mean().reset_index()

    if cluster_summary['Recency'].isnull().all():
        return []

    highest_recency_cluster = int(cluster_summary.loc[cluster_summary['Recency'].idxmax(), 'Cluster'])
    max_recency_stockcodes = rfm_df[rfm_df['Cluster'] == highest_recency_cluster]['StockCode'].astype(str).tolist()

    product_names = ds[['StockCode', 'Description']].drop_duplicates()
    max_recency_products = product_names[product_names['StockCode'].astype(str).isin(max_recency_stockcodes)]

    return max_recency_products.fillna("").to_dict(orient="records")

#Finds products from the cluster with the highest purchase frequency. Helps in identifying fast-moving or frequently reordered products.
def get_high_frequency_prod(ds: pd.DataFrame, rfm_df: pd.DataFrame) -> list:
    rfm_df = rfm_df.replace([np.inf, -np.inf], np.nan).dropna()
    cluster_summary = rfm_df.groupby('Cluster').agg({'Frequency': 'mean'}).reset_index()

    if cluster_summary['Frequency'].isnull().all():
        return []  

    highest_frequency_cluster = int(cluster_summary.loc[cluster_summary['Frequency'].idxmax(), 'Cluster'])
    high_freq_stockcodes = rfm_df[rfm_df['Cluster'] == highest_frequency_cluster]['StockCode'].astype(str).tolist()

    product_names = ds[['StockCode', 'Description']].drop_duplicates()
    high_freq_products = product_names[product_names['StockCode'].astype(str).isin(high_freq_stockcodes)]

    return high_freq_products.fillna("").to_dict(orient="records")

# Retrieves products from the cluster with the highest monetary value (total spending). Useful for detecting high-value or premium products.
def get_high_monetary_prod(ds: pd.DataFrame, rfm_df: pd.DataFrame) -> list:
    rfm_df = rfm_df.replace([np.inf, -np.inf], np.nan).dropna()
    cluster_summary = rfm_df.groupby('Cluster').agg({'Monetary': 'mean'}).reset_index()

    if cluster_summary['Monetary'].isnull().all():
        return []

    highest_monetary_cluster = int(cluster_summary.loc[cluster_summary['Monetary'].idxmax(), 'Cluster'])
    high_monetary_stockcodes = rfm_df[rfm_df['Cluster'] == highest_monetary_cluster]['StockCode'].astype(str).tolist()

    product_names = ds[['StockCode', 'Description']].drop_duplicates()
    high_monetary_products = product_names[product_names['StockCode'].astype(str).isin(high_monetary_stockcodes)]

    return high_monetary_products.fillna("").to_dict(orient="records")

# highlighting popular and frequently purchased products.
def get_high_recency_high_frequency_prod(ds: pd.DataFrame, rfm_df: pd.DataFrame) -> list:
    rfm_df = rfm_df.replace([np.inf, -np.inf], np.nan).dropna()
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean'
    }).reset_index()

    if cluster_summary[['Recency', 'Frequency']].isnull().all().any():
        return []

    highest_recency_cluster = int(cluster_summary.loc[cluster_summary['Recency'].idxmax(), 'Cluster'])
    highest_frequency_cluster = int(cluster_summary.loc[cluster_summary['Frequency'].idxmax(), 'Cluster'])

    high_recency_stockcodes = rfm_df[rfm_df['Cluster'] == highest_recency_cluster]['StockCode'].astype(str).tolist()
    high_frequency_stockcodes = rfm_df[rfm_df['Cluster'] == highest_frequency_cluster]['StockCode'].astype(str).tolist()

    high_demand_stockcodes = list(set(high_recency_stockcodes) & set(high_frequency_stockcodes)) 

    product_names = ds[['StockCode', 'Description']].drop_duplicates()
    high_demand_products = product_names[product_names['StockCode'].astype(str).isin(high_demand_stockcodes)]

    return high_demand_products.fillna("").to_dict(orient="records")

#identifying high-value, high-demand products.
def get_high_amount_high_frequency_prod(ds: pd.DataFrame, rfm_df: pd.DataFrame) -> list:
    rfm_df = rfm_df.replace([np.inf, -np.inf], np.nan).dropna()
    cluster_summary = rfm_df.groupby('Cluster').agg({'Frequency': 'mean', 'Monetary': 'mean'}).reset_index()

    if cluster_summary[['Frequency', 'Monetary']].isnull().all().any():
        return []

    highest_frequency_cluster = int(cluster_summary.loc[cluster_summary['Frequency'].idxmax(), 'Cluster'])
    highest_amount_cluster = int(cluster_summary.loc[cluster_summary['Monetary'].idxmax(), 'Cluster'])

    high_freq_stockcodes = rfm_df[rfm_df['Cluster'] == highest_frequency_cluster]['StockCode'].astype(str).tolist()
    high_amount_stockcodes = rfm_df[rfm_df['Cluster'] == highest_amount_cluster]['StockCode'].astype(str).tolist()

    # Merge both clusters
    high_value_stockcodes = list(set(high_freq_stockcodes + high_amount_stockcodes))

    product_names = ds[['StockCode', 'Description']].drop_duplicates()
    high_value_products = product_names[product_names['StockCode'].astype(str).isin(high_value_stockcodes)]

    return high_value_products.fillna("").to_dict(orient="records")

# detecting slow-moving or underperforming products.
def get_low_frequency_low_monetary_prod(ds: pd.DataFrame, rfm_df: pd.DataFrame) -> list:
    rfm_df = rfm_df.replace([np.inf, -np.inf], np.nan).dropna()
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()

    if cluster_summary[['Frequency', 'Monetary']].isnull().all().any():
        return []

    lowest_frequency_cluster = int(cluster_summary.loc[cluster_summary['Frequency'].idxmin(), 'Cluster'])
    lowest_monetary_cluster = int(cluster_summary.loc[cluster_summary['Monetary'].idxmin(), 'Cluster'])

    low_freq_stockcodes = rfm_df[rfm_df['Cluster'] == lowest_frequency_cluster]['StockCode'].astype(str).tolist()
    low_monetary_stockcodes = rfm_df[rfm_df['Cluster'] == lowest_monetary_cluster]['StockCode'].astype(str).tolist()

    low_value_stockcodes = list(set(low_freq_stockcodes + low_monetary_stockcodes))
    
    product_names = ds[['StockCode', 'Description']].drop_duplicates()
    low_value_products = product_names[product_names['StockCode'].astype(str).isin(low_value_stockcodes)]

    return low_value_products.fillna("").to_dict(orient="records")

# Useful for detecting outdated or declining products.
def get_low_recency_low_frequency_prod(ds: pd.DataFrame, rfm_df: pd.DataFrame) -> list:
    rfm_df = rfm_df.replace([np.inf, -np.inf], np.nan).dropna()
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean'
    }).reset_index()

    if cluster_summary[['Recency', 'Frequency']].isnull().all().any():
        return []

    lowest_recency_cluster = int(cluster_summary.loc[cluster_summary['Recency'].idxmin(), 'Cluster'])
    lowest_frequency_cluster = int(cluster_summary.loc[cluster_summary['Frequency'].idxmin(), 'Cluster'])

    low_recency_stockcodes = rfm_df[rfm_df['Cluster'] == lowest_recency_cluster]['StockCode'].astype(str).tolist()
    low_frequency_stockcodes = rfm_df[rfm_df['Cluster'] == lowest_frequency_cluster]['StockCode'].astype(str).tolist()

    outdated_stockcodes = list(set(low_recency_stockcodes) & set(low_frequency_stockcodes))  # Intersection

    product_names = ds[['StockCode', 'Description']].drop_duplicates()
    outdated_products = product_names[product_names['StockCode'].astype(str).isin(outdated_stockcodes)]

    return outdated_products.fillna("").to_dict(orient="records")

#Useful for identifying staple or frequently re-ordered items.
def get_high_loyalty_products(ds: pd.DataFrame, rfm_df: pd.DataFrame) -> list:
    rfm_df = rfm_df.replace([np.inf, -np.inf], np.nan).dropna()
    cluster_summary = rfm_df.groupby('Cluster').agg({'Frequency': 'mean'}).reset_index()

    if cluster_summary['Frequency'].isnull().all():
        return []

    highest_loyalty_cluster = int(cluster_summary.loc[cluster_summary['Frequency'].idxmax(), 'Cluster'])
    high_loyalty_stockcodes = rfm_df[rfm_df['Cluster'] == highest_loyalty_cluster]['StockCode'].astype(str).tolist()

    product_names = ds[['StockCode', 'Description']].drop_duplicates()
    loyal_products = product_names[product_names['StockCode'].astype(str).isin(high_loyalty_stockcodes)]

    return loyal_products.fillna("").to_dict(orient="records")

#detecting price-sensitive items that attract budget-conscious customers.
def get_price_sensitive_products(ds: pd.DataFrame, rfm_df: pd.DataFrame) -> list:
    rfm_df = rfm_df.replace([np.inf, -np.inf], np.nan).dropna()
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()

    if cluster_summary[['Frequency', 'Monetary']].isnull().all().any():
        return []

    highest_frequency_cluster = int(cluster_summary.loc[cluster_summary['Frequency'].idxmax(), 'Cluster'])
    lowest_monetary_cluster = int(cluster_summary.loc[cluster_summary['Monetary'].idxmin(), 'Cluster'])

    frequent_stockcodes = rfm_df[rfm_df['Cluster'] == highest_frequency_cluster]['StockCode'].astype(str).tolist()
    low_value_stockcodes = rfm_df[rfm_df['Cluster'] == lowest_monetary_cluster]['StockCode'].astype(str).tolist()

    price_sensitive_stockcodes = list(set(frequent_stockcodes) & set(low_value_stockcodes))  # Intersection

    product_names = ds[['StockCode', 'Description']].drop_duplicates()
    price_sensitive_products = product_names[product_names['StockCode'].astype(str).isin(price_sensitive_stockcodes)]

    return price_sensitive_products.fillna("").to_dict(orient="records")

#high-value products
def get_potential_high_value_products(ds: pd.DataFrame, rfm_df: pd.DataFrame) -> list:
    rfm_df = rfm_df.replace([np.inf, -np.inf], np.nan).dropna()
    if rfm_df.empty or rfm_df['Cluster'].isnull().all():
        return []
    rfm_df['StockCode'] = rfm_df['StockCode'].astype(str)
    cluster_summary = rfm_df.groupby('Cluster').agg({'Recency': 'mean', 'Monetary': 'mean'}).reset_index()
    if cluster_summary.shape[0] < 2 or cluster_summary[['Recency', 'Monetary']].isnull().all().any():
        return []
    lowest_recency_cluster = int(cluster_summary.loc[cluster_summary['Recency'].idxmin(), 'Cluster'])
    highest_monetary_cluster = int(cluster_summary.loc[cluster_summary['Monetary'].idxmax(), 'Cluster'])
    low_recency_stockcodes = set(rfm_df[rfm_df['Cluster'] == lowest_recency_cluster]['StockCode'])
    high_monetary_stockcodes = set(rfm_df[rfm_df['Cluster'] == highest_monetary_cluster]['StockCode'])
    premium_stockcodes = list(low_recency_stockcodes & high_monetary_stockcodes)
    product_names = ds[['StockCode', 'Description']].drop_duplicates()
    premium_products = product_names[product_names['StockCode'].astype(str).isin(premium_stockcodes)]
    return premium_products.fillna("").to_dict(orient="records")