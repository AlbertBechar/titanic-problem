# Titanic Survival Prediction 🚢

An end-to-end machine learning pipeline for the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. The goal is to predict which passengers survived the sinking of the Titanic using features such as age, sex, passenger class, fare and family size.

The project covers the full workflow: data inspection, missing-value imputation, feature engineering, encoding, model training with a Random Forest, hyperparameter tuning, and generating a Kaggle-ready submission file.

---

## 📁 Repository Structure

```
.
├── data/
│   ├── train.csv          # Kaggle training data (891 passengers, with labels)
│   └── test.csv           # Kaggle test data (418 passengers, no labels)
├── titanic.ipynb          # Main notebook with the full pipeline
├── submission.csv         # Generated predictions for Kaggle (created by the notebook)
└── README.md
```

> **Note:** The data files are not included in this repository. Download `train.csv` and `test.csv` from the [competition data page](https://www.kaggle.com/competitions/titanic/data) and place them in a `data/` folder.

---

## ⚙️ Pipeline Overview

### 1. Data Loading
Train and test sets are loaded and temporarily concatenated so that every preprocessing and encoding step is applied identically to both.

### 2. Missing Value Imputation

| Feature    | Missing | Strategy |
|------------|---------|----------|
| `Age`      | 263     | Median age of the passenger's class (`Pclass`) |
| `Fare`     | 1       | Median fare of 3rd-class passengers who embarked at Southampton |
| `Embarked` | 2       | Most common port among 1st-class passengers with a similar fare (~80) |
| `Cabin`    | 1014    | Filled with `M` (Missing); the first letter is kept as the deck |

### 3. Feature Engineering

| Feature      | Description |
|--------------|-------------|
| `Title`      | Extracted from the passenger name and grouped (Mr, Mrs, Miss/Ms, Master, Dr, Rev, Military, Nobility) |
| `Cabin`      | Deck letter (A–G) or `M` if unknown |
| `Age_Bin`    | Age grouped into 10-year bins |
| `Fare_Bin`   | Fare per person (fare ÷ number of passengers on the same ticket), binned |
| `Num_Family` | `SibSp + Parch + 1`, the passenger's total family size on board |

Raw columns that were replaced by engineered features (`Name`, `Age`, `Ticket`, `Fare`) are dropped.

### 4. Encoding
- Ordinal / binned features are label-encoded.
- Nominal features (`Sex`, `Embarked`, `Title`) are one-hot encoded.

### 5. Modeling
- **Model:** `RandomForestClassifier` (scikit-learn)
- **Validation:** 80/20 train/validation split
- **Tuning:** `RandomizedSearchCV` for a broad search, followed by `GridSearchCV` around the best parameters

### 6. Submission
The tuned model is refit on the full training set, predictions are made on the test set, and a `submission.csv` file with `PassengerId` and `Survived` columns is written for upload to Kaggle.

---

## 📊 Results

| Model | Validation Accuracy |
|-------|---------------------|
| Baseline Random Forest | 80.4% |
| Random Forest + RandomizedSearchCV | 81.6% |
| Kaggle public leaderboard | _add your score here_ |

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```
3. Download the data from Kaggle into `data/`.
4. Open and run the notebook:
   ```bash
   jupyter notebook titanic.ipynb
   ```
5. Upload the generated `submission.csv` on the [competition submission page](https://www.kaggle.com/competitions/titanic/submissions).

Tested with Python 3.9.

---

## 🔮 Possible Improvements

- Impute missing ages using `Title` and `Pclass` together instead of class alone
- Try gradient boosting models (XGBoost, LightGBM, CatBoost) and compare with Random Forest
- Use k-fold cross-validation on the full training set instead of a single hold-out split
- Combine several models in a voting or stacking ensemble
- Add survival-rate features for families / ticket groups
- Inspect feature importances to remove redundant features

---

## 🛠️ Tech Stack

Python · pandas · NumPy · Matplotlib · scikit-learn · Jupyter
