# AI-Powered Retail Sales Forecasting Dashboard 📈

A complete end-to-end Machine Learning solution for retail sales prediction using **Facebook Prophet**. This dashboard provides interactive visualizations, trend analysis, and performance metrics to help businesses plan ahead.

## 🚀 Features

- **Automated Data Cleaning**: Formats dates, handles missing values, and prepares data for time-series modeling.
- **Advanced Forecasting**: Predicts sales for 7, 14, 30, 60, or 90 days.
- **Dynamic Regressors**: Optionally include Stocks and Price to improve prediction accuracy.
- **Interactive Visuals**: Switch between Line, Area, and Bar charts.
- **Trend Decomposition**: View weekly and yearly seasonality patterns.
- **Performance Evaluation**: Detailed metrics including MAE, RMSE, and MAPE.
- **Export Options**: Download forecast results as CSV or a full PDF report.
- **Modern UI**: Polished Streamlit interface with animations and interactive KPIs.

## 🛠️ Project Structure

```
Future Interns/
├── app.py              # Main Streamlit Dashboard
├── requirements.txt    # Project dependencies
├── data/               # Folder for datasets
│   └── retail_sales.csv # Sample retail dataset
├── models/             # (Optional) Saved models
├── assets/             # Images and animations
└── README.md           # Instructions
```

## 📦 Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset Requirement
The application expects a CSV file with at least the following columns:
- `Date`: Format (YYYY-MM-DD)
- `Sales`: Historical sales figures
- `Stocks` (Optional): Inventory levels
- `Price` (Optional): Unit price

## 🤖 Model Information
The core forecasting engine uses **Facebook Prophet**, an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality.

---
**Created by Aman Shaikh**
