import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diabetic_data.csv')
df.head()

cols = ['weight', 'payer_code',]
df.drop(columns=cols, inplace=True)
df.replace('?',np.nan, inplace=True)

for col in df.select_dtypes(include='object'):
    df[col] = df[col].fillna(df[col].mode()[0])


df.sort_values(by=["patient_nbr","encounter_id"], inplace = True)
df = df.drop_duplicates(subset=['patient_nbr'], keep='last')

for col in df.columns:
    if col not in ["readmitted_label", "diag_1", "diag_2", "diag_3"]:
        df[col] = pd.to_numeric(df[col], errors='ignore')


def simplify_diag(code):
    try:
        code = float(code)
    except:
        return "Unknown"

    if 390 <= code <= 459 or code == 785:
        return "Circulatory"
    elif 460 <= code <= 519 or code == 786:
        return "Respiratory"
    elif 520 <= code <= 579 or code == 787:
        return "Digestive"
    elif 250 <= code <= 251:
        return "Diabetes"
    elif 800 <= code <= 999:
        return "Injury"
    elif 710 <= code <= 739:
        return "Musculoskeletal"
    elif 580 <= code <= 629 or code == 788:
        return "Genitourinary"
    else:
        return "Other"

for col in ["diag_1", "diag_2", "diag_3"]:
    df[col] = df[col].apply(simplify_diag)

df["readmitted_label"] = df["readmitted"].map({
    "NO": 0,
    ">30": 1,
    "<30": 2
})

replace_map = {"No":0, "Steady":1, "Up":2, "Down":3}

med_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone',
    'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]

for col in med_cols:
    df[col] = df[col].map(replace_map)

df = pd.get_dummies(df, drop_first=True)

num_cols = df.select_dtypes(include=['int64','float64']).columns


cols_for_outliers = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

for col in cols_for_outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]

df.to_csv("diabetic_data_cleaned.csv", index=False)

print("Clean dataset saved as diabetic_data_cleaned.csv")
