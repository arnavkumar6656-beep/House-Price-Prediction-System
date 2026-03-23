# House Price Prediction using XGBoost

A machine learning project that predicts California housing prices using the XGBoost regression algorithm.

---

## Overview

This project loads the California Housing dataset, performs exploratory data analysis, trains an XGBoost regression model, and evaluates its performance on both training and test data.

---

## Dataset

**California Housing Dataset** — loaded directly from `sklearn.datasets.fetch_california_housing()`

| Property | Detail |
|---|---|
| Source | scikit-learn built-in dataset |
| Features | 8 numeric features (MedInc, HouseAge, AveRooms, etc.) |
| Target | Median house price (in units of $100,000) |
| Size | 20,640 samples |

The target column is added to the DataFrame as `price`.

---

## Project Structure

```
house_price_prediction/
│
├── house_price_prediction.ipynb   # Main Jupyter notebook
└── README.md                      # This file
```

---

## Requirements

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Correlation heatmap visualization |
| `scikit-learn` | Dataset, train/test split, metrics |
| `xgboost` | XGBRegressor model |

---

## Workflow

### 1. Load Data
```python
house_price_dataset = sklearn.datasets.fetch_california_housing()
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
house_price_dataframe['price'] = house_price_dataset.target
```

### 2. Exploratory Data Analysis
- `isnull().sum()` — check for missing values
- `describe()` — summary statistics
- Correlation heatmap using seaborn

### 3. Prepare Features
```python
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```

### 4. Train Model
```python
model = XGBRegressor()
model.fit(X_train, Y_train)
```

### 5. Evaluate
Predictions are made on both training and test sets and evaluated using R² and MAE.

---

## Results

| Split | R² Score | Mean Absolute Error |
|---|---|---|
| Training | 0.9437 | 0.1934 |
| Test | 0.8338 | 0.3109 |

The gap between training R² (0.94) and test R² (0.83) indicates mild overfitting. The model generalizes reasonably well but could be improved with hyperparameter tuning.

---

## Visualization

A scatter plot of **Actual Prices vs Predicted Prices** on the training set is generated to visually assess model fit. Points closer to a diagonal line indicate better predictions.

---

## Possible Improvements

- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV` (tune `n_estimators`, `max_depth`, `learning_rate`)
- Cross-validation for more robust performance estimates
- Feature importance analysis using `model.feature_importances_`
- Regularization (`reg_alpha`, `reg_lambda`) to reduce the train/test gap
