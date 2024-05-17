# House Prices Prediction

The aim of this project is to develop a machine learning model to predict house sale prices based on various features. The model aims to accurately estimate house prices using advanced regression techniques and thorough data preprocessing methods.

## Features of the Code

- **Data Preprocessing**: Utilizes imputation for handling missing values and one-hot encoding for converting categorical variables to numerical format.
- **Model Selection**: Implements a `RandomForestRegressor` as the core model for prediction, chosen for its ability to handle complex, non-linear relationships.
- **Hyperparameter Tuning**: Employs `GridSearchCV` with cross-validation to optimize hyperparameters, ensuring the model's performance is robust and well-tuned.
- **Model Evaluation**: Uses the Root Mean Squared Error (RMSE) metric to evaluate the model's prediction accuracy on a validation set.
- **Pipeline Integration**: Constructs a streamlined machine learning pipeline with `Pipeline` and `ColumnTransformer` to ensure a seamless and reproducible workflow.

## Data Visualization

The feature importance plot below shows the top 10 features influencing the house price predictions made by the RandomForestRegressor model:

![Figure_1](https://github.com/Ayan0209/HousePricing/assets/33597664/04da7489-d83d-4a57-9be2-17145438d9e6)

