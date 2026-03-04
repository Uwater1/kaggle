import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ydf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix

# Settings
%matplotlib inline
plt.style.use('fivethirtyeight')
sns.set_palette("viridis")
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def clean_and_feature(df, target_name=None):
    """Local function for custom automated data cleaning and feature engineering."""
    df = df.copy()
    
    # 1. Missing value indicators
    for col in df.columns:
        if df[col].isnull().any():
            df[f"{col}_isMissing"] = df[col].isnull().astype(int)
    
    # 2. Drop standard ID and high-text columns
    drops = [c for c in ['PassengerId', 'Name', 'Id', 'id', 'Ticket'] if c in df.columns]
    if len(drops) > 0:
        df = df.drop(columns=drops)

    # 3. Handle mixed types in categorical columns
    cat_check_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in cat_check_cols:
        if col != target_name:
            df[col] = df[col].map(lambda x: str(x) if pd.notnull(x) else x)
            
    # 4. Drop high-cardinality metadata
    obj_drops = []
    total_rows = len(df)
    for col in df.select_dtypes(include=['object']).columns:
        if col == target_name: continue
        num_unique = df[col].nunique()
        if num_unique > 0.6 * total_rows:
            obj_drops.append(col)
            
    if len(obj_drops) > 0:
        df = df.drop(columns=obj_drops)
        
    return df

# Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
target_name = 'SalePrice' # Adjust for your competition
is_classification = False # Adjust for your competition

# Preprocess
train_clean = clean_and_feature(train_df, target_name)
X_train, X_val, y_train, y_val = train_test_split(train_clean, train_clean[target_name], test_size=0.2, random_state=42)

# --- YDF GradientBoostedTreesLearner ---
print(f"🚀 Training YDF GradientBoostedTreesLearner on {target_name}...")
model = ydf.GradientBoostedTreesLearner(label=target_name, 
                                        num_trees=500,
                                        max_depth=6).train(X_train)

print(f"✅ Training finished!")
model.describe()

# Evaluation
y_pred = model.predict(X_val)
if is_classification:
    acc = accuracy_score(y_val, y_pred)
    print(f"🎯 Validation Accuracy: {acc:.2%}")
else:
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"🏠 RMSE: ${rmse:,.2f}")
    print(f"📉 MAE: ${mae:,.2f}")
    print(f"📈 R2 Score: {r2:.4f}")

# Submission
if test_df is not None:
    test_clean = clean_and_feature(test_df)
    predictions = model.predict(test_clean)
    final_predictions = predictions.flatten() if hasattr(predictions, 'flatten') else predictions
    
    submission_file = pd.DataFrame({
        "Id": test_df.iloc[:, 0], 
        target_name: final_predictions
    })
    submission_file.to_csv('submission_updated.csv', index=False)
    print("🎉 'submission_updated.csv' is ready.")
