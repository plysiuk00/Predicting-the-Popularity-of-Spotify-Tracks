import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Import for visualization
import seaborn as sns # Import for visualization

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# For handling class imbalance (SMOTE)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# --- DATA LOADING AND CLEANING ---
# Load dataset and drop any rows with missing values
data = pd.read_csv('data/dataset.csv')
data = data.dropna()

# --- TARGET VARIABLE CREATION AND FEATURE ENGINEERING ---

# 1. Create the binary target variable (1 if popularity > 50, 0 otherwise)
data['popularity_flag'] = 0
data.loc[data['popularity'] > 50, 'popularity_flag'] = 1

# 2. Convert explicit column from boolean to integer (1 or 0) if needed
if data['explicit'].dtype == 'bool':
    data['explicit'] = data['explicit'].astype(int)

# 3. Create new ratio features to capture complex interactions (Feature Engineering)
epsilon = 1e-6 # Small constant to prevent division by zero
data['energy_acoustic_ratio'] = data['energy'] / (data['acousticness'] + epsilon)
data['loudness_instrumental_ratio'] = data['loudness'] / (data['instrumentalness'] + epsilon)


# 4. Apply One-Hot Encoding to 'track_genre' (Crucial for model performance)
data_encoded = pd.get_dummies(data, columns=['track_genre'], prefix='genre', drop_first=True)


# --- FEATURE SELECTION (X) AND TARGET (y) DEFINITION ---

# Columns to drop: IDs, names, artists, and the original continuous 'popularity' score
cols_to_drop = [
    'Unnamed: 0', 'track_id', 'artists', 'album_name',
    'track_name', 'popularity', 'popularity_flag'
]

# X includes all numerical features, ratio features, and encoded genres
X = data_encoded.drop(cols_to_drop, axis=1, errors='ignore').select_dtypes(include=np.number)
# y is the binary target variable
y = data_encoded['popularity_flag']

# --- DATA SPLITTING ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# --- MODEL PIPELINE DEFINITION ---

# Initialize preprocessors and resampler
smote = SMOTE(random_state=1)
scaler = StandardScaler()

# Define models using ImbPipeline to ensure scaling and SMOTE only run on training data
models = []
models.append(('Random Forest (Scaled + SMOTE)', ImbPipeline(steps=[('scaler', scaler), ('smote', smote), ('clf', RandomForestClassifier(random_state=1))])))
models.append(('Decision Tree (Scaled + SMOTE)', ImbPipeline(steps=[('scaler', scaler), ('smote', smote), ('clf', DecisionTreeClassifier(random_state=1))])))
# Note: XGBoost is generally less sensitive to scaling but we include it for consistency
models.append(('XGB Classifier (Scaled + SMOTE)', ImbPipeline(steps=[('scaler', scaler), ('smote', smote), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1))])))

# --- TRAINING AND EVALUATION ---
print("--- Classification Results (with Scaler, SMOTE, and Enhanced Features) ---")
for name, pipeline in models:
    # Train the pipeline (Scaler and SMOTE are applied only to X_train/y_train)
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    predictions = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    # Print results, focusing on Accuracy and Flag 1 (Popular) metrics
    print(f'Accuracy for {name}: {accuracy:.3f}')
    print(f'Precision (Flag 1): {report["1"]["precision"]:.3f}')
    print(f'Recall (Flag 1):    {report["1"]["recall"]:.3f}')
    print("-" * 30)