#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import joblib


# In[4]:


# Load the datasets
train_data = pd.read_csv('/Users/naanahope/Downloads/male_players (legacy).csv',low_memory=False)
test_data = pd.read_csv('/Users/naanahope/Downloads/players_22.csv',low_memory=False)


# In[5]:


# Inspect column names
print(train_data.columns)


# In[6]:


# Display the first few rows to confirm the column names
print(train_data.head())


# In[7]:


# Plot histogram for the 'overall' column
sns.histplot(train_data['overall'])
plt.show()


# In[8]:


# List of columns to be removed if they exist
irrelevant_columns = ['player_id', 'player_url', 'short_name', 'long_name', 'player_face_url', 'fifa_version', 'fifa_update', 'fifa_update_date']

# Drop irrelevant columns from train_data if they exist
train_data = train_data.drop(columns=[col for col in irrelevant_columns if col in train_data.columns])

# Drop irrelevant columns from test_data if they exist
test_data = test_data.drop(columns=[col for col in irrelevant_columns if col in test_data.columns])


# In[9]:


# Align test_data with train_data columns
test_data = test_data.reindex(columns=train_data.columns)
test_data


# In[10]:


# Fill missing values only for numeric columns
numeric_cols = train_data.select_dtypes(include=['number']).columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())
test_data[numeric_cols] = test_data[numeric_cols].fillna(train_data[numeric_cols].mean())


# In[11]:


# Display the cleaned datasets
print(train_data.head())
print(test_data.head())


# In[12]:


# Select only numeric columns for correlation calculation
numeric_cols = train_data.select_dtypes(include=[np.number]).columns
numeric_train_data = train_data[numeric_cols]

# Calculate correlation matrix
correlation_matrix = numeric_train_data.corr()

# Select features with high correlation to 'overall'
high_corr_features = correlation_matrix.index[abs(correlation_matrix['overall']) > 0.5]
print(high_corr_features)

# Create a new dataset with these features
train_data_high_corr = train_data[high_corr_features]
test_data_high_corr = test_data[high_corr_features]


# In[13]:


# Split data into features and target
X_train = train_data_high_corr.drop(columns=['overall'])
y_train = train_data_high_corr['overall']
X_test = test_data_high_corr.drop(columns=['overall'])
y_test = test_data_high_corr['overall']

# Initialize models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'GradientBoost': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Evaluate models with cross-validation
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f'{name} CV RMSE: {np.sqrt(-cv_scores).mean()}')

# Train and evaluate the best model
best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print(f'Test MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')


# In[14]:


# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [3, 5, 7]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best CV RMSE: {np.sqrt(-grid_search.best_score_)}')


# In[15]:


# Assume X_train, y_train are already defined from your dataset
# Initialize and train your model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file named 'best_model.pkl'
joblib.dump(model, 'best_model.pkl')



# In[ ]:





# In[ ]:





# In[ ]:




