# Housing Price Prediction

## Project Overview
This is a beginner-friendly project for predicting housing prices using Python, Pandas, Matplotlib, Seaborn, and scikit-learn.  
It covers the full workflow of a machine learning project: data loading, cleaning, visualization, feature engineering, model training, and evaluation.

## Dataset
- Kaggle Housing Dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)  
- Features include house size, number of bedrooms, garage, year built, etc.  
- Target variable: `SalePrice`

## Features Used
- `OverallQual` – Overall material and finish quality  
- `GrLivArea` – Above ground living area in square feet  
- `GarageCars` – Size of garage in car capacity  
- `TotalBsmtSF` – Total square feet of basement area  
- `FullBath` – Full bathrooms above grade  
- `YearBuilt` – Original construction year  
- `TotRmsAbvGrd` – Total rooms above grade (excluding bathrooms)

## Models Implemented
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  

## Evaluation Metrics
- RMSE (Root Mean Squared Error)  
- R² (Coefficient of Determination)  

## Usage
1. Download `train.csv` from Kaggle and put it in the same folder.  
2. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
