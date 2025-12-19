import pandas as pd

def clean_and_save():
    df = pd.read_csv('data/retail_sales.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.dropna()
    df.to_csv('data/cleaned_retail_sales.csv', index=False)
    print("Cleaned dataset saved to 'data/cleaned_retail_sales.csv'.")

if __name__ == "__main__":
    clean_and_save()
