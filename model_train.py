import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("diabetic_data_cleaned.csv")
print("After load:", df.shape)
print("NaN in target:", df['readmitted_label'].isna().sum())

df = df.dropna(subset=['readmitted_label'])
print("After dropping NaN targets:", df.shape)

df = df.drop(columns=[
    'readmitted_NO',
    'readmitted_>30',
    'encounter_id',
    'patient_nbr'
])

y = df["readmitted_label"]
X = df.drop(columns=['readmitted_label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_param_dist = {
    'n_estimators': randint(100, 600),
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None], 
    'class_weight': ['balanced', 'balanced_subsample']
}

rf = RandomForestClassifier(random_state=42)

rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_param_dist,
    n_iter=10,             # number of random combinations
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

rf_random_search.fit(X_train, y_train)

print("Best RF Parameters:", rf_random_search.best_params_)
print("Best RF Score:", rf_random_search.best_score_)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest Accuracy: ", accuracy_score(y_test, rf_pred))
print("Train accuracy: ", accuracy_score(y_train, rf.predict(X_train)))
print("\nConfusion Matrix: \n",confusion_matrix(y_test, rf_pred))
print("\nClassification Report: \n",classification_report(y_test,rf_pred))

xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,subsample=0.8,
                    colsample_bytree=0.8, random_state=42, tree_method='hist')

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("XGBoost Accuracy: ", accuracy_score(y_test, xgb_pred))
print("Train accuracy: ", accuracy_score(y_train, xgb.predict(X_train)))
print("\nConfusion Matrix: \n",confusion_matrix(y_test, xgb_pred))
print("\nClassification Report: \n",classification_report(y_test,xgb_pred))

print("Model Comparison: ")
print("Random Forest Accuracy: ", accuracy_score(y_test, rf_pred))
print("XGBoost Accuracy: ", accuracy_score(y_test, xgb_pred))

rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

plt.figure(figsize=(6,4))
plt.bar(['Random Forest', 'XGBoost'], [rf_acc, xgb_acc])
plt.ylim(0,1.05)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.grid(axis='y', linewidth=0.3, linestyle=':')
plt.tight_layout()
plt.show()

rf_cm = confusion_matrix(y_test, rf_pred)
xgb_cm = confusion_matrix(y_test, xgb_pred)
labels = sorted(y.unique())

fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.heatmap(rf_cm, annot=True, fmt='d', ax=axes[0], cbar=False)
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual'); axes[0].set_title('Random Forest CM')
axes[0].set_xticklabels(labels); axes[0].set_yticklabels(labels, rotation=0)

sns.heatmap(xgb_cm, annot=True, fmt='d', ax=axes[1], cbar=False)
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual'); axes[1].set_title('XGBoost CM')
axes[1].set_xticklabels(labels); axes[1].set_yticklabels(labels, rotation=0)

plt.tight_layout()
plt.show()

rf_imp = rf.feature_importances_
xgb_imp = xgb.feature_importances_
cols = np.array(X.columns)

rf_top_idx = np.argsort(rf_imp)[-20:]
xgb_top_idx = np.argsort(xgb_imp)[-20:]

fig, axes = plt.subplots(1,2, figsize=(14,8))
# RF horizontal bar
axes[0].barh(range(len(rf_top_idx)), rf_imp[rf_top_idx])
axes[0].set_yticks(range(len(rf_top_idx)))
axes[0].set_yticklabels(cols[rf_top_idx])
axes[0].set_title('Random Forest - Top 20 Importances')

# XGB horizontal bar
axes[1].barh(range(len(xgb_top_idx)), xgb_imp[xgb_top_idx])
axes[1].set_yticks(range(len(xgb_top_idx)))
axes[1].set_yticklabels(cols[xgb_top_idx])
axes[1].set_title('XGBoost - Top 20 Importances')

plt.tight_layout()
plt.show()




