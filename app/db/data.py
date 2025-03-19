import pandas as pd
path1 = "./data/combine_dataset_2009_2011.csv"
ds=pd.read_csv(path1)
ds = ds.replace('', pd.NA)
ds = ds.dropna()    
ds = ds.reset_index(drop=True)
