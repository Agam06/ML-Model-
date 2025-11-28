# ML-Model-I

Diabetic Patient Readmission Prediction

A Machine Learning project using Random Forest, XGBoost, and Hyperparameter Optimization

📌 Project Overview
This project predicts hospital readmission of diabetic patients based on clinical, demographic, and treatment-related features using supervised machine learning.
The dataset used is the Diabetes 130-US Hospitals (1999–2008) dataset from the UCI Machine Learning Repository.
The model predicts whether a patient will be:
0 → NO readmission
1 → Readmitted after >30 days
2 → Readmitted within <30 days
The final pipeline includes:
 Complete data preprocessing
 Data leakage removal
 Feature engineering
 Train–test split
 Scaling
 Random Forest & XGBoost model training
 RandomizedSearchCV hyperparameter tuning
 Visualizations using Matplotlib & Seaborn

🛠️ Technologies Used
 Python
 Pandas, NumPy
 Scikit-learn
 XGBoost
 Seaborn, Matplotlib
 RandomizedSearchCV

📂 Dataset
Original dataset:
Diabetes 130-US Hospitals for Years 1999–2008
(UCI Machine Learning Repository)
Contains:
 100,000+ hospital encounters
 50+ features
 Missing values encoded as "?"
 High-cardinality categorical data
 ICD-9 diagnosis codes

🔧 Preprocessing Steps
✔ Replace missing values
The dataset uses "?" instead of NaN.
Replaced with np.nan and filled:
Categorical: mode
Numerical: mostly clean

✔ Remove irrelevant/leaky columns
Removed columns that cause data leakage or have no predictive value:
encounter_id
patient_nbr
One-hot encoded columns:
readmitted_NO
readmitted_>30
readmitted_<30
weight (96% missing)
payer_code

✔ Map target values
Mapped readmission categories:
NO  → 0  
>30 → 1  
<30 → 2

✔ Simplify diagnosis codes
Converted ICD-9 codes to broad disease groups:
Circulatory
Respiratory
Digestive
Diabetes
Injury
Musculoskeletal
Genitourinary
Other

✔ Encode medications
Mapped:
No, Steady, Up, Down → 0, 1, 2, 3

✔ One-hot encoding
Converted categorical features using:
pd.get_dummies(drop_first=True)

✔ Scaling
Applied StandardScaler after train-test split.

🤖 Modeling
Two classifiers were built:
1. Random Forest Classifier
Baseline model
Bagging-based ensemble
Handles large feature spaces well

2. XGBoost Classifier
Boosting-based model
Handles imbalance + nonlinear patterns
More powerful and accurate

🎯 Hyperparameter Tuning (RandomizedSearchCV)
For Random Forest:
rf_param_dist = {
    'n_estimators': randint(100, 600),
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample']
}

For XGBoost:
xgb_param_dist = {
    'n_estimators': randint(200, 600),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 12),
    'min_child_weight': randint(1, 10),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 5)
}

Performed 3-fold cross-validation with 10–20 iterations.

📊 Visualizations
The following graphs were generated:
✔ Accuracy Comparison
Bar chart comparing Random Forest vs XGBoost.
✔ Confusion Matrices
Seaborn heatmaps showing multi-class predictions.
✔ Feature Importances
Top 20 features displayed for both models.
✔ ROC Curve (Multi-Class)
One-vs-rest ROC curves for all 3 classes.

🧪 Results
Model	Train Accuracy	Test Accuracy
Random Forest	~92–98%	~82–86%
XGBoost	~90–96%	~83–87%
(Scores vary by tuned parameters)
XGBoost generally performed slightly better on unseen data.

📁 Project Structure
│── diabetic_data_cleaned.csv
│── preprocessing.py
│── model_training.py
│── tuning_random_forest.py
│── tuning_xgboost.py
│── visualizations.py
│── README.md

🚀 How to Run the Project
1. Install dependencies
pip install -r requirements.txt
2. Run preprocessing
python preprocessing.py
3. Train models
python model_training.py
4. Run hyperparameter tuning
python tuning_random_forest.py
python tuning_xgboost.py
5. Generate visualizations
python visualizations.py

📌 Conclusion
This project demonstrates how a real medical dataset can be cleaned, engineered, and modeled using advanced ML techniques. With proper preprocessing and careful removal of data leakage, both Random Forest and XGBoost provide strong predictive performance for hospital readmission.

🤝 Contributions
Feel free to fork and enhance the project. Pull requests are welcome.


Just tell me!
