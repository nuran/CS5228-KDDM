import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Separate features and target variable from training data
X = train_data.drop(columns=['price'])  # Replace 'price' with the actual target column name if different
y = train_data['price']  # Replace 'price' with the actual target column name if different

# Step 1: Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.get_dummies(X_train, drop_first=True)
X_valid = pd.get_dummies(X_valid, drop_first=True)

# Align columns to ensure compatibility
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)


# Step 4: Create preprocessing pipelines for both numeric and categorical features
# numerical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# Step 5: Combine preprocessing steps into a column transformer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, train)
#     ]
# )

# Step 6: Create a pipeline that first preprocesses the data and then applies the XGBoost model
model_pipeline = Pipeline(steps=[
    ('model', XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=20, random_state=42))
])

# Step 3: Fit the model pipeline to the training data

print(X_train)
model_pipeline.fit(X_train, y_train)

# Step 4: Make predictions on the validation set
y_pred = model_pipeline.predict(X_valid)

# Step 5: Evaluate the model's performance
mae = mean_absolute_error(y_valid, y_pred)
mse = mean_squared_error(y_valid, y_pred)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
r2 = r2_score(y_valid, y_pred)

# Print the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R2): {r2:.2f}')
