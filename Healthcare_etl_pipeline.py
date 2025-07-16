# healthcare_etl_pipeline.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# Step 1: Load the healthcare dataset
# Sample healthcare data 
data = {
    'Patient_ID': [101, 102, 103, 104, 105],
    'Age': [29, 47, 35, np.nan, 52],
    'Gender': ['Male', 'Female', 'Female', 'Male', np.nan],
    'Blood_Pressure': [120, 140, np.nan, 130, 110],
    'Cholesterol': ['High', 'Normal', 'High', np.nan, 'Normal'],
    'Smoker': ['Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
df.to_csv("raw_healthcare_data.csv", index=False)
print("Raw Data:")
print(df)


# Step 2: Define column types
# Drop Patient_ID (not useful for modeling)
df = df.drop(columns=['Patient_ID'])

# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Step 3: Create Preprocessing Pipelines
# Numerical data pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical data pipeline
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])


# Step 4: Build Full ETL Pipeline
etl_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Step 5: Fit and Transform the Data
processed_data = etl_pipeline.fit_transform(df)

# Step 6: Convert to DataFrame
# Get feature names after transformation
cat_feature_names = etl_pipeline.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
all_feature_names = num_cols + cat_feature_names.tolist()

processed_df = pd.DataFrame(processed_data.toarray() if hasattr(processed_data, 'toarray') else processed_data,columns=all_feature_names)

# Step 7: Save the Processed Data
processed_df.to_csv("processed_healthcare_data.csv", index=False)
print("\nProcessed Data:")
print(processed_df.head())
print("\n✅ ETL Pipeline completed. Processed data saved to 'processed_healthcare_data.csv'")
