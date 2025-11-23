import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualization
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
print("Libraries imported and visualization style set.")

# --- 1. LOAD DATA ---

# Load dataset from the specified path
df = pd.read_csv("data/dataset.csv")
print(f"Data loaded successfully. Total rows: {len(df)}")

# Remove redundant/useless column (often created during saving/loading)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("'Unnamed: 0' column removed.")


# --- 2. INITIAL DATA INSPECTION ---

print("\n--- Initial Data Head ---")
print(df.head())

print("\n--- Data Types ---")
print(df.info())

# --- 3. DISTRIBUTION OF TARGET VARIABLE (POPULARITY) ---

# Visualize the distribution of the target variable ('popularity')
plt.figure(figsize=(10, 5))
sns.histplot(df["popularity"], bins=30, kde=True)
plt.title("Distribution of Track Popularity")
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.show()
# Observation: The distribution is heavily skewed towards low popularity values (0-20),
# indicating a significant class imbalance between popular and unpopular tracks.


# --- 4. CORRELATION ANALYSIS (Feature Relationships) ---

# Select only numerical features for correlation calculation
# Note: 'explicit' is included here if it's stored as int/float (0/1)
numeric_features = df.select_dtypes(include=['float64', 'int64'])

# Generate a correlation heatmap
plt.figure(figsize=(14, 10))
# Setting annot=False to keep the map clean, but we could set it to True to display values.
sns.heatmap(numeric_features.corr(), annot=False, cmap="viridis")
plt.title("Correlation Heatmap of Audio Features")
plt.show()
# Observation: Look for strong positive or negative correlations with the 'popularity' row/column.
# Strong candidates are 'loudness', 'energy', 'acousticness', and 'instrumentalness'.


# --- 5. TOP CATEGORICAL FEATURE ANALYSIS (Genre) ---

# Group by 'track_genre' and calculate the mean popularity for the top 20 genres
top_genres = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(20)

# Visualize the top 20 genres by average popularity
plt.figure(figsize=(12, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='rocket')
plt.title("Top-20 Genres by Average Popularity")
plt.xlabel("Average Popularity")
plt.ylabel("Genre")
plt.show()
# Observation: Genres like 'pop-film' and 'k-pop' show significantly higher average popularity,
# confirming that 'track_genre' is a crucial feature for the classification task.


# --- 6. INDIVIDUAL FEATURE VISUALIZATION VS POPULARITY ---

# List of features to visualize against popularity
features = ['duration_ms','explicit','danceability', 'energy','key','loudness','mode','speechiness', 'acousticness', 'instrumentalness','liveness','valence', 'tempo','time_signature']

# Create scatter plots for each feature against 'popularity'
plt.figure(figsize=(18, 18))
for i, feature in enumerate(features, 1):
    plt.subplot(5, 3, i)
    sns.scatterplot(x=df[feature], y=df['popularity'], alpha=0.3)
    plt.title(f'{feature} vs Popularity')
    plt.xlabel(feature)
    plt.ylabel('Popularity')

plt.tight_layout()
plt.show()
# Observation: Scatter plots help visualize the relationships identified in the heatmap.
# For example, 'loudness' shows a clear positive trend with popularity.
# 'Danceability' shows a weaker, but still noticeable, positive trend.