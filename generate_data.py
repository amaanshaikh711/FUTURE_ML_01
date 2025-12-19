import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data():
    np.random.seed(42)
    start_date = datetime(2022, 1, 1)
    num_days = 730
    dates = [start_date + timedelta(days=x) for x in range(num_days)]
    
    # Categories and Regions for richer visualization
    categories = ['Electronics', 'Clothing', 'Furniture', 'Home Decor', 'Accessories']
    regions = ['Maharashtra', 'Madhya Pradesh', 'Uttar Pradesh', 'Delhi', 'Karnataka']
    payment_modes = ['UPI', 'Credit Card', 'Debit Card', 'COD', 'EMI']
    
    # Generate base sales with seasonality and trend
    base_sales = 500
    trend = np.linspace(0, 300, num_days)
    seasonal_weekly = 70 * np.sin(2 * np.pi * np.array(range(num_days)) / 7)
    seasonal_yearly = 150 * np.sin(2 * np.pi * np.array(range(num_days)) / 365)
    noise = np.random.normal(0, 40, num_days)
    
    sales = base_sales + trend + seasonal_weekly + seasonal_yearly + noise
    sales = np.maximum(sales, 50) # Ensure no unrealistic sales
    
    # Stocks (correlated with sales but with lag)
    stocks = 1200 - (sales * 0.6) + np.random.normal(0, 60, num_days)
    stocks = np.maximum(stocks, 100)
    
    # Price (some fluctuations)
    price = 25 + np.sin(np.array(range(num_days)) / 45) + np.random.normal(0, 0.8, num_days)
    
    # Create the dataframe
    data_list = []
    for i in range(num_days):
        # We'll simulate multiple transactions per day or just assign attributes to the daily total for dashboard flair
        data_list.append({
            'Date': dates[i],
            'Sales': round(sales[i], 2),
            'Stocks': int(stocks[i]),
            'Price': round(price[i], 2),
            'Category': np.random.choice(categories, p=[0.2, 0.4, 0.15, 0.1, 0.15]),
            'Region': np.random.choice(regions, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
            'Payment_Mode': np.random.choice(payment_modes, p=[0.3, 0.2, 0.15, 0.25, 0.1])
        })
    
    df = pd.DataFrame(data_list)
    df.to_csv('data/retail_sales.csv', index=False)
    print("Enhanced sample dataset 'data/retail_sales.csv' generated.")

if __name__ == "__main__":
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    generate_sample_data()
