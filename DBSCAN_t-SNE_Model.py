import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

# === Suppress Warnings ===
warnings.filterwarnings('ignore')

# === Set Default Font for Unicode Emojis Compatibility ===
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Compatible with emoji

# === Load Dataset ===
df = pd.read_csv('cleaned_dataset.csv')  # Adjust path if needed

# === Encode Categorical Columns ===
label_encoders = {}
categorical_cols = ['Komuna', 'Niveli Akademik', 'Mosha', 'Gjinia']
for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# === Feature Selection ===
features = ['Komuna_encoded', 'Viti Akademik', 'Niveli Akademik_encoded',
            'Mosha_encoded', 'Gjinia_encoded', 'Numri i nxenesve']
X = df[features]

# === Standard Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === t-SNE for Dimensionality Reduction ===
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)
df['TSNE-1'] = X_tsne[:, 0]
df['TSNE-2'] = X_tsne[:, 1]

# === DBSCAN Clustering ===
dbscan = DBSCAN(eps=1.2, min_samples=10)
clusters = dbscan.fit_predict(X_scaled)
df['Cluster'] = clusters

# === Count of Points per Cluster ===
print("\n📊 Cluster sizes (including -1 = outliers):")
print(df['Cluster'].value_counts().sort_index())

# === Cluster Breakdown by Komuna ===
cluster_city = df.groupby(['Cluster', 'Komuna']).size().unstack(fill_value=0)
print("\n🏙️ Cluster Breakdown by Komuna:")
print(cluster_city)

# === t-SNE Scatter Plot with DBSCAN Clusters ===
plt.figure(figsize=(12, 8))
palette = sns.color_palette("hsv", len(set(clusters)))
sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2', hue='Cluster', palette=palette, s=60, edgecolor='w', linewidth=0.4)
plt.title("🔍 DBSCAN Clusters in 2D (t-SNE View)", fontsize=14)
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.legend(title='Cluster', loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Optional: Save processed file with cluster info ===
df.to_csv("clustered_dataset_dbscan.csv", index=False)
print("✅ Clustered dataset saved as 'clustered_dataset_dbscan.csv'")