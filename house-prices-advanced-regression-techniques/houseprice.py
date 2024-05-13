import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')

# Separate target from predictors and apply logarithmic transformation
y = np.log(train_data['SalePrice'])
X = train_data.drop(['SalePrice'], axis=1)

# Select numerical columns
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

# Select categorical columns with relatively low cardinality (convenient choice here)
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object" and X[cname].nunique() < 10]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestRegressor(random_state=0)

# Create a grid of parameters to search
param_grid = {
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [None, 10, 20, 30]
}

# Create and evaluate the pipeline with grid search
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=3)

# Split data into train and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Fit the grid search
print("Starting Grid Search...")
grid_search.fit(X_train, y_train)
print("Grid Search Complete.")
# Best model from grid search
best_model = grid_search.best_estimator_

# Preprocessing of validation data, get predictions
preds = best_model.predict(X_valid)

# Convert predictions back from log scale
preds_exp = np.exp(preds)

# Calculate RMSE on the original scale
score = mean_squared_error(np.exp(y_valid), preds_exp, squared=False)
print('Validation RMSE:', score)

# Generate test predictions
test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
final_predictions = np.exp(best_model.predict(test_data))

# Save predictions in the format used by sample_submission.csv
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': final_predictions})
output.to_csv('submission.csv', index=False)
