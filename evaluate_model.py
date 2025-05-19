import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('yt performance analytics.csv')
except FileNotFoundError:
    print("Error: 'yt performance analytics.csv' not found in the project folder.")
    raise

# Load the trained model pipeline (includes scaler and model)
try:
    pipeline = pickle.load(open('revenue_model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: revenue_model.pkl not found.")
    raise

# Define features and target
features = ['Video Duration', 'Views', 'Likes', 'Shares', 'New Subscribers', 'Video Thumbnail CTR (%)']
target = 'Estimated Revenue (USD)'

# Check for missing columns
missing_features = [col for col in features if col not in df.columns]
if missing_features:
    print(f"Error: Missing features in dataset: {missing_features}")
    raise KeyError(f"Missing features: {missing_features}")
if target not in df.columns:
    print(f"Error: Target column '{target}' not found in dataset.")
    raise KeyError(f"Missing target: {target}")

# Handle missing values
df[features] = df[features].fillna(df[features].median())
df[target] = df[target].fillna(df[target].median())

# Feature Engineering: Add interaction terms
df['Views_Likes_Interaction'] = df['Views'] * df['Likes']
df['Views_CTR_Interaction'] = df['Views'] * df['Video Thumbnail CTR (%)']

# Update features list
features.extend(['Views_Likes_Interaction', 'Views_CTR_Interaction'])

# Prepare X and y
X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print results
print(f"R² Score: {r2:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} USD")
