# Used Car Price Drivers: Regression and Classification Modeling

## 1. Project Objective

This project answers the business question:

> **What drives the price of a used car?**

The goal is not only to build predictive models, but also to interpret which vehicle characteristics most strongly influence price. The project uses:

- Exploratory Data Analysis (EDA)
- Regression modeling
- Classification modeling
- Cross-validation
- Hyperparameter tuning
- Feature importance analysis

---

## 2. Executive Summary 

A concise summary of key findings is available here:

[Executive Summary](reports/executive_summary.md)

### What drives the price of a car?

Based on exploratory analysis and machine learning evidence, the strongest drivers of used car price are:

1. **Vehicle year / age**  
   Newer cars generally have higher prices due to less wear, newer technology, and stronger resale value.

2. **Odometer mileage**  
   Higher mileage decreases price because it signals greater usage and depreciation.

3. **Manufacturer / brand**  
   Brand reputation, reliability, luxury positioning, and resale demand influence price.

4. **Condition**  
   Vehicles in excellent or like-new condition command higher prices than damaged or salvage vehicles.

5. **Vehicle characteristics**  
   Fuel type, transmission, drivetrain, and body type influence pricing through utility and demand.

6. **Location / state**  
   Regional differences affect pricing due to supply-demand variation.

### Key Insight

Car pricing is **not random**. It is systematically driven by:

> **age, mileage, brand value, and condition**, with additional influence from vehicle characteristics and regional demand.

---

## 3. Dataset

Place the dataset at:

```text
data/vehicles.csv
```

Expected columns include:

- `price`
- `year`
- `manufacturer`
- `model`
- `condition`
- `fuel`
- `odometer`
- `transmission`
- `drive`
- `type`
- `state`

Only available columns are used dynamically.

---

## 4. Project Structure

```text
used_car_price_project_enhanced/
│
├── config/
│   └── config.yaml
│
├── data/
│   └── vehicles.csv
│
├── logs/
│   └── project.log
│
├── models/
│   ├── best_regression_model.joblib
│   ├── tuned_best_regression_model.joblib
│   └── best_classification_model.joblib
│
├── plots/
│   ├── price_distribution.png
│   ├── log_price_distribution.png
│   ├── price_vs_odometer.png
│   ├── price_by_year.png
│   ├── price_by_manufacturer.png
│   ├── price_by_condition.png
│   ├── price_by_fuel.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
├── reports/
│   ├── data_summary.csv
│   ├── missing_values.csv
│   ├── manufacturer_price_summary.csv
│   ├── condition_price_summary.csv
│   ├── regression_model_results.csv
│   ├── classification_model_results.csv
│   ├── feature_importance.csv
│   ├── best_hyperparameters.txt
│   └── executive_summary.md
│
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── reporting.py
│   └── logger.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 5. How to Run

### Recommended Environment

Use Python 3.10 or 3.11 for better package compatibility.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

If using Anaconda directly:

```bash
C:\Users\indmi_q2tg30l\anaconda3\python.exe main.py
```

---

## 6. Modeling Approach

### Regression Models

The regression task predicts actual vehicle price.

Models included:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Regression metrics:

- MAE
- MSE
- RMSE
- R²
- Cross-validated RMSE mean
- Cross-validated RMSE standard deviation

---

### Classification Models

The classification task predicts whether a car is expensive.

A vehicle is labeled expensive if:

```text
price > median(price)
```

This creates a binary target:

```text
0 = Not Expensive
1 = Expensive
```

Models included:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross-validated F1 mean
- Cross-validated F1 standard deviation

---

## 7. Classification Model Interpretation

To complement regression analysis, a classification model was developed to categorize vehicles into two groups:

- **Expensive**: above median price
- **Not Expensive**: below or equal to median price

### Key Observations

The confusion matrix and classification report help evaluate how well the model separates expensive and non-expensive vehicles.

The model correctly identifies many vehicles in both classes:

- True negatives: vehicles correctly classified as not expensive
- True positives: vehicles correctly classified as expensive
- False positives: lower-priced vehicles incorrectly classified as expensive
- False negatives: higher-priced vehicles incorrectly classified as not expensive

Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.89      0.88     21609
           1       0.89      0.87      0.88     21512

    accuracy                           0.88     43121
   macro avg       0.88      0.88      0.88     43121
weighted avg       0.88      0.88      0.88     43121

### Interpretation

The classification model demonstrates balanced performance across both classes, indicating that the available vehicle features contain meaningful signals for separating high-value and low-value vehicles.

### Deeper Insight

The classification results show that price is not arbitrary. The data contains a clear separation boundary between expensive and non-expensive cars.

This confirms that:

- Price is not random
- Price is systematically determined by key vehicle characteristics
- The same features that help predict actual price also help classify price category

### Connection to Price Drivers

The classification model reinforces the importance of:

- Vehicle age / year
- Mileage / odometer
- Manufacturer / brand
- Condition
- Fuel type
- Transmission
- Vehicle type

### Critical Insight

The classification model confirms that car pricing is driven by a combination of **age, usage, brand value, and condition**, creating a strong and predictable separation between high-value and low-value vehicles.

### Business Interpretation

From a dealership perspective:

- Newer, low-mileage, premium-brand vehicles should be priced higher
- Older or high-mileage vehicles should be priced more competitively
- Inventory quality and brand positioning directly influence pricing strategy
- Classification can help segment inventory into higher-value and lower-value groups

---

## 8. Cross-Validation Approach

Cross-validation is used to estimate how well each model generalizes beyond a single train-test split.

The project uses:

```text
cv_folds: 3
```

For regression:

- Models are trained and evaluated across multiple folds
- The main cross-validation metric is RMSE
- Lower RMSE indicates stronger predictive performance

For classification:

- Models are trained and evaluated across multiple folds
- The main cross-validation metric is F1 score
- Higher F1 indicates better balance between precision and recall

Because the full dataset contains roughly 350,000 rows, cross-validation can be computationally expensive. The configuration file includes a development sample option to allow faster iteration.

---

## 9. Hyperparameter Tuning

The project uses `RandomizedSearchCV` to improve model performance.

If XGBoost is available, tuning can be performed on XGBoost. Otherwise, tuning may fall back to Random Forest.

Tuned hyperparameters may include:

- Number of estimators / trees
- Maximum tree depth
- Learning rate
- Subsample ratio
- Column sample ratio
- Minimum samples for splitting

The best parameters are saved to:

```text
reports/best_hyperparameters.txt
```

The tuned model is saved to:

```text
models/tuned_best_regression_model.joblib
```

---

## 10. Model Assumptions

### Linear Regression

- Assumes a linear relationship between features and price
- Assumes independent observations
- Assumes residuals are approximately normally distributed
- Assumes constant variance of residuals
- Sensitive to outliers and multicollinearity

### Ridge Regression

- Uses the same broad assumptions as linear regression
- Adds L2 regularization to reduce coefficient instability
- Useful when many encoded categorical variables create correlated predictors

### Lasso Regression

- Uses the same broad assumptions as linear regression
- Adds L1 regularization
- Can shrink some coefficients to zero
- Can act as a feature-selection method
- May underperform if many weak but useful predictors exist

### Random Forest Regressor / Classifier

- Does not require linear relationships
- Captures nonlinear effects and feature interactions
- Assumes training data is representative of future data
- Can be computationally expensive on large datasets
- Feature importance can be biased toward high-cardinality variables

### Gradient Boosting Regressor

- Builds trees sequentially
- Each tree attempts to correct the errors of previous trees
- Captures nonlinear relationships
- Can overfit if too many trees or excessive depth are used
- Usually benefits from careful tuning

### XGBoost Regressor / Classifier

- Efficient gradient boosting algorithm
- Strong at capturing nonlinear effects and interactions
- Requires tuning of tree depth, learning rate, number of estimators, and sampling parameters
- Can overfit without regularization and validation

### Logistic Regression

- Used here for classification, not direct price prediction
- Predicts whether a car is above the median price
- Assumes a linear relationship between features and the log-odds of the class
- Benefits from standardized numeric features
- Provides an interpretable baseline for classification

---

## 11. Data Assumptions

- The dataset is assumed to be representative of the used car market
- Extreme prices are removed to reduce the influence of likely errors or unusual listings
- Missing categorical values are treated as `unknown`
- Rare categorical levels are grouped as `other` to reduce dimensionality
- Price is assumed to be meaningfully related to vehicle characteristics such as year, mileage, brand, and condition
- Regional and macroeconomic factors are not fully captured
- Listing descriptions and seller behavior are not included unless available in structured fields

---

## 12. Limitations

- The dataset may contain noisy or inconsistent seller-entered values
- Vehicle trim, accident history, service history, and local market demand may not be fully represented
- Model results depend on the quality of available structured columns
- Seller behavior, negotiation, and local market shocks are not fully captured
- A sample may be used for faster development; full-data training can be enabled in `config/config.yaml`

---

## 13. Future Work

Future improvements could include:

- NLP analysis of listing descriptions
- Regional economic variables
- SHAP values for deeper explainability
- Full-data training compared with sampled-data training
- Interactive dashboard for business interpretation
- More detailed trim-level and vehicle-history features

---

## 14. Final Conclusion

The analysis consistently shows that used car prices are driven primarily by:

- Vehicle age / year
- Mileage / odometer
- Manufacturer / brand
- Condition
- Vehicle type and characteristics
- Location / state

This conclusion is supported by:

- Exploratory plots
- Summary tables
- Regression model results
- Classification model results
- Feature importance from tree-based models

Tree-based models generally outperform purely linear models, indicating that used car pricing is governed by nonlinear interactions between features rather than by one isolated variable.

The agreement between exploratory analysis, regression modeling, and classification modeling strengthens confidence in the findings and confirms that car pricing is predictable, explainable, and driven by structured vehicle attributes.

---

## 15. Main Business Takeaway

For dealerships, sellers, or pricing analysts, the most important pricing levers are:

1. Prioritize vehicle age and mileage when setting baseline price
2. Adjust price based on manufacturer and model reputation
3. Apply premiums for better condition
4. Account for body type, drivetrain, transmission, and fuel type
5. Consider regional supply-demand differences

In summary:

> **Used car prices are primarily driven by age, mileage, brand, condition, and vehicle characteristics, with nonlinear interactions between these factors shaping final market value.
Model-output importance plots show that vehicle year, odometer mileage, manufacturer, condition, and vehicle type are the strongest predictors of used car price. Permutation importance further confirms that these variables materially affect prediction accuracy, meaning they are not just correlated with price but actively drive model performance.**