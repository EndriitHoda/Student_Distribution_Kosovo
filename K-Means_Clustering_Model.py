import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# 1. === Load Dataset ===
df = pd.read_csv("cleaned_dataset.csv")  # Use your dataset file

# 2. === Encode Categorical Columns ===
label_encoders = {}
for col in ['Komuna', 'Niveli Akademik', 'Mosha', 'Gjinia']:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. === Prepare Features for Clustering ===
X = df[['Komuna_encoded', 'Viti Akademik', 'Niveli Akademik_encoded', 'Mosha_encoded', 'Gjinia_encoded', 'Numri i nxenesve']]

# 4. === Scale the Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. === Apply K-Means Clustering ===
k = 3  # Based on your elbow method
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. === PCA for Visualization (2D) ===
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

# 7. === Visualize Clusters ===
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=60)
plt.title("🔍 K-Means Clusters (PCA View)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# 8. === Cluster Breakdown by Komuna ===
komuna_breakdown = df.groupby(['Cluster', 'Komuna']).size().unstack(fill_value=0)
print("📊 Cluster Breakdown by Komuna:\n", komuna_breakdown)

# 9. === Breakdown by Mosha ===
mosha_counts = df.groupby(['Cluster', 'Mosha']).size().unstack().fillna(0)
mosha_counts.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='Pastel1')
plt.title("👶 Mosha Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Students")
plt.legend(title="Mosha")
plt.tight_layout()
plt.show()

# 10. === Breakdown by Niveli Akademik ===
niveli_counts = df.groupby(['Cluster', 'Niveli Akademik']).size().unstack().fillna(0)
niveli_counts.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='Pastel2')
plt.title("🎓 Niveli Akademik Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Students")
plt.legend(title="Niveli Akademik")
plt.tight_layout()
plt.show()

# 11. === Breakdown by Gjinia ===
gjinia_counts = df.groupby(['Cluster', 'Gjinia']).size().unstack().fillna(0)
gjinia_counts.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='Set3')
plt.title("⚧️ Gjinia Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Students")
plt.legend(title="Gjinia")
plt.tight_layout()
plt.show()