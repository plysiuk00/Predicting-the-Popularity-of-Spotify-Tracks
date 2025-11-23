import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

df = pd.read_csv('data/dataset.csv')

df['popularity_flag'] = 0
df.loc[df['popularity'] > 50, 'popularity_flag'] = 1

cols_to_drop = [
    'Unnamed: 0', 'track_id', 'artists', 'album_name',
    'track_name', 'popularity', 'track_genre', 'popularity_flag'
]
X = df.drop(cols_to_drop, axis=1, errors='ignore').select_dtypes(include=np.number)

y = df['popularity_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

models = []
models.append(('Random Forest Classifier', RandomForestClassifier(random_state=1)))
models.append(('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=1)))
models.append(('XGB Classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1)))
models.append(('Decision Tree Classifier', DecisionTreeClassifier(random_state=1)))

print("--- Результати Класифікації ---")
for name, model in models:
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    print(f'Accuracy for {name}: {accuracy:.3f}')
    print(f'Precision (Flag 1): {report["1"]["precision"]:.3f}')
    print(f'Recall (Flag 1):    {report["1"]["recall"]:.3f}')
    print("-" * 30)
