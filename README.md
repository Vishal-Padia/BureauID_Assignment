# BureauID_Assignment

## 1. Approach Taken

### Data Loading
The script begins by loading the training and test datasets using the `load_data` function. The datasets are expected to be in CSV format and are read into pandas DataFrames.

### Data Preprocessing
The `preprocess_data` function is responsible for preparing the data for model training. This involves:
- **Separating Features and Target**: The target variable (`Application Status`) is separated from the features.
- **Identifying Column Types**: Numeric and categorical columns are identified based on their data types.
- **Creating a Preprocessor**: A `ColumnTransformer` is used to handle both numeric and categorical features. For numeric features, missing values are imputed using the median strategy, and the data is scaled using `StandardScaler`. For categorical features, missing values are imputed with a constant value ("missing"), and the data is one-hot encoded using `OneHotEncoder`.

### Model Training
The `train_model` function creates a machine learning pipeline that includes the preprocessor and a `RandomForestClassifier`. The model is trained on the preprocessed training data.

### Making Predictions
The `make_predictions` function prepares the test data (removing the `UID` column if present) and uses the trained model to generate predictions. The predictions are saved to a CSV file named `predictions.csv`.

### Main Execution
The `main` function orchestrates the entire process, from loading the data to making predictions and evaluating the model on the training data.

## 2. Insights and Conclusions from Data

### Data Insights
- **Missing Values**: The preprocessing step handles missing values in both numeric and categorical features. Numeric features are imputed using the median, while categorical features are imputed with a constant value.
- **Feature Scaling**: Scaling numeric features ensures that no single feature dominates the model due to its scale.
- **One-Hot Encoding**: Categorical features are one-hot encoded to convert them into a format suitable for machine learning models.

### Model Insights
- **High Accuracy**: The Random Forest Classifier achieved near-perfect accuracy on the training dataset, indicating that the model is likely overfitting to the training data.
- **Precision and Recall**: Both precision and recall are 1.00 for both classes, suggesting that the model is performing exceptionally well on the training data but may not generalize well to unseen data.

## 3. Performance on Train Dataset

### Metrics
- **Training Accuracy**: 0.9999
- **Classification Report**:

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| APPROVED    | 1.00      | 1.00   | 1.00     | 6677    |
| DECLINED    | 1.00      | 1.00   | 1.00     | 3323    |
| **Accuracy**|           |        | **1.00** | **10000**|
| **Macro Avg**| 1.00     | 1.00   | 1.00     | 10000   |
| **Weighted Avg**| 1.00  | 1.00   | 1.00     | 10000   |

### Interpretation
- **High Accuracy**: The model achieved a training accuracy of 99.99%, which is extremely high and suggests that the model is fitting the training data very well.
- **Precision and Recall**: Both precision and recall are 1.00 for both classes, indicating that the model is correctly identifying all instances of both `APPROVED` and `DECLINED` applications without any false positives or false negatives.
- **Potential Overfitting**: The near-perfect performance on the training data raises concerns about overfitting. The model may not generalize well to unseen data, and further evaluation on a validation or test set is necessary to confirm its real-world performance.

### Recommendations
- **Cross-Validation**: Perform cross-validation to get a more robust estimate of the model's performance.
- **Hyperparameter Tuning**: Fine-tune the hyperparameters of the Random Forest Classifier to potentially improve generalization.
- **Feature Engineering**: Explore additional feature engineering techniques to improve model performance.
- **Model Evaluation on Test Data**: Evaluate the model on the test dataset to assess its real-world performance and generalization capabilities.