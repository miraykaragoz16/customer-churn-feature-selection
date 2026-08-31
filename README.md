# Customer Churn Prediction with Feature Selection

This project investigates how different feature selection methods affect customer churn prediction in the telecommunications sector. The study compares filter, wrapper, and embedded feature selection techniques and evaluates several machine learning models using the IBM Telco Customer Churn dataset.

## Project Objective

The main objectives of this project are to:

- identify the most influential factors associated with customer churn;
- reduce redundant and low-impact features;
- compare multiple feature selection approaches;
- evaluate classification models using the selected feature set; and
- interpret the final model with SHAP.

## Dataset

The project uses the **IBM Telco Customer Churn** dataset, which contains information about 7,043 telecommunications customers and 21 original variables. These variables include customer demographics, subscribed services, contract details, payment methods, monthly charges, total charges, and churn status.

After converting `TotalCharges` to a numeric variable and removing missing observations, 7,032 records remained for analysis.

> The dataset file is not included in this repository. To run the notebook, download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it in the same directory as the notebook.

## Project Workflow

1. Exploratory data analysis
2. Data cleaning and preprocessing
3. Encoding categorical variables
4. Train-test split
5. Class balancing with SMOTE
6. Feature selection
7. Classification model comparison
8. XGBoost hyperparameter tuning and threshold optimization
9. Model interpretation with SHAP

## Feature Selection Methods

Three main feature selection approaches were examined:

### Filter Methods

- Chi-Square
- ANOVA F-test
- Mutual Information

### Wrapper Method

- Recursive Feature Elimination with Cross-Validation (RFECV)

### Embedded Method

- LASSO Logistic Regression

Following correlation-based redundancy removal and LASSO selection, the original 30-feature encoded space was reduced to **18 final features** while preserving most of the predictive performance.

## Machine Learning Models

The following classification algorithms were compared:

- Logistic Regression
- Random Forest
- XGBoost

The models were trained on the SMOTE-balanced training set and evaluated on the original, imbalanced test set.

## Results

### Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7321 | 0.4974 | 0.7647 | 0.6027 | 0.8324 |
| Random Forest | 0.7768 | 0.5909 | 0.5214 | 0.5540 | 0.8052 |
| XGBoost | **0.7882** | **0.5964** | 0.6283 | **0.6120** | **0.8357** |

XGBoost achieved the strongest overall baseline performance and was selected for hyperparameter optimization.

### Optimized XGBoost

RandomizedSearchCV was applied using 5-fold cross-validation. The optimized model reached:

- ROC-AUC: **0.8344**
- Recall: **0.8021** at the default decision threshold
- Best decision threshold: **0.63**
- F1-Score at the optimized threshold: **0.6282**
- Recall at the optimized threshold: **0.6979**
- Precision at the optimized threshold: **0.5711**

The optimized model provides a more useful balance for churn management, where identifying customers at risk of leaving is especially important.

## Key Findings

The analysis identified contract type, customer tenure, internet service type, and payment method as major factors in churn prediction. In particular:

- two-year and one-year contracts were strongly associated with customer retention;
- longer tenure reduced the predicted risk of churn;
- fiber-optic internet service was associated with higher churn risk; and
- electronic check payments were associated with higher churn risk.

SHAP analysis was used to explain the global importance and directional impact of the selected variables on the final XGBoost predictions.

## Technologies and Libraries

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- XGBoost
- SHAP

## Installation

Install the required Python libraries with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
```

## Running the Project

1. Clone or download this repository.
2. Place the dataset CSV file in the project directory.
3. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
4. Run the cells sequentially from beginning to end.

## Project File

The full data analysis, feature selection steps, model training, evaluation, and SHAP interpretation are available in [`eda - Kopya.ipynb`](./eda%20-%20Kopya.ipynb).

## Author

**Zekiye Miray Karagöz**  
Mathematical Engineering, Yıldız Technical University

## License

This project was prepared for academic purposes.
