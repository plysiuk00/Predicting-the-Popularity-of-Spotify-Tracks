import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/dataset.csv")

# Remove useless column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

plt.figure(figsize=(10, 5))
sns.histplot(df["popularity"], bins=30, kde=True)
plt.title("Distribution of Track Popularity")
plt.xlabel("Popularity")
plt.ylabel("Count")
# plt.show()

numeric_features = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(14, 10))
sns.heatmap(numeric_features.corr(), annot=False, cmap="viridis")
plt.title("Correlation Heatmap of Audio Features")
# plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["danceability"], y=df["popularity"], alpha=0.3)
plt.title("Danceability vs Popularity")
plt.xlabel("Danceability")
plt.ylabel("Popularity")
# plt.show()

top_genres = df.groupby("track_genre")["popularity"].mean().sort_values(ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title("Top-20 Genres by Average Popularity")
plt.xlabel("Average Popularity")
plt.ylabel("Genre")
# plt.show()

# Visualize numerical features
df.hist(bins=20, figsize=(15, 10))
plt.show()

# List of features to visualize against popularity
features = ['duration_ms','explicit','danceability', 'energy','key','loudness','mode','speechiness', 'acousticness', 'instrumentalness','liveness','valence', 'tempo','time_signature']

# Create scatter plots
plt.figure(figsize=(15, 15))
for i, feature in enumerate(features, 1):
    plt.subplot(5, 3, i)
    sns.scatterplot(x=df[feature], y=df['popularity'])
    plt.title(f'{feature} vs Popularity')
    plt.xlabel(feature)
    plt.ylabel('Popularity')

plt.tight_layout()
plt.show()

