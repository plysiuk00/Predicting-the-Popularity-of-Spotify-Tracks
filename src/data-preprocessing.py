import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler # <--- НОВИЙ ІМПОРТ
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline as ImbPipeline 
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv('data/dataset.csv')
data = data.dropna() 


data['popularity_flag'] = 0
data.loc[data['popularity'] > 50, 'popularity_flag'] = 1
if data['explicit'].dtype == 'bool':
    data['explicit'] = data['explicit'].astype(int)

epsilon = 1e-6
data['energy_acoustic_ratio'] = data['energy'] / (data['acousticness'] + epsilon)
data['loudness_instrumental_ratio'] = data['loudness'] / (data['instrumentalness'] + epsilon)


data_encoded = pd.get_dummies(data, columns=['track_genre'], prefix='genre', drop_first=True)


cols_to_drop = [
    'Unnamed: 0', 'track_id', 'artists', 'album_name', 
    'track_name', 'popularity', 'popularity_flag'
]

X = data_encoded.drop(cols_to_drop, axis=1, errors='ignore').select_dtypes(include=np.number)
y = data_encoded['popularity_flag']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

smote = SMOTE(random_state=1)
scaler = StandardScaler()

models = []
models.append(('Random Forest (Scaled + SMOTE)', ImbPipeline(steps=[('scaler', scaler), ('smote', smote), ('clf', RandomForestClassifier(random_state=1))])))
models.append(('Decision Tree (Scaled + SMOTE)', ImbPipeline(steps=[('scaler', scaler), ('smote', smote), ('clf', DecisionTreeClassifier(random_state=1))])))
models.append(('XGB Classifier (Scaled + SMOTE)', ImbPipeline(steps=[('scaler', scaler), ('smote', smote), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1))])))


print("--- Результати Класифікації (Зі Scaler, SMOTE та Комбінованими Ознаками) ---")
for name, pipeline in models:
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    print(f'Accuracy for {name}: {accuracy:.3f}')
    print(f'Precision (Flag 1): {report["1"]["precision"]:.3f}')
    print(f'Recall (Flag 1):    {report["1"]["recall"]:.3f}')
    print("-" * 30)