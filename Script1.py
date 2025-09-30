import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

st.title("🛍️ Mall Customer Segmentation with K-Means")


st.subheader("📊 Dataset Preview")
st.write(df.head())


X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


st.sidebar.header("⚙️ Controls")
max_k = st.sidebar.slider("Max k for Elbow/Silhouette", 2, 10, 10)
n_clusters = st.sidebar.slider("Select number of clusters (k)", 2, 10, 5)


st.subheader("📉 Elbow Method")
wcss = []
for i in range(1, max_k+1):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, max_k+1), wcss, marker="o")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Method for Optimal k")
st.pyplot(fig)


st.subheader("📏 Silhouette Analysis")
silhouette_scores = []
for i in range(2, max_k+1):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

fig, ax = plt.subplots()
ax.plot(range(2, max_k+1), silhouette_scores, marker="o", color="green")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Analysis")
st.pyplot(fig)



st.subheader(f"🎯 K-Means Clustering Results (k={n_clusters})")
kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans_final.fit_predict(X_scaled)
df['Cluster'] = labels

# Scatter Plot
fig, ax = plt.subplots()
sns.scatterplot(
    x='Annual Income (k$)', y='Spending Score (1-100)',
    hue='Cluster', data=df, palette='Set1', ax=ax, s=80
)
centers = scaler.inverse_transform(kmeans_final.cluster_centers_)
ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.6, marker='X', label='Centroids')
ax.set_title("Customer Segmentation")
ax.legend()
st.pyplot(fig)


# Average Spending per Cluster

st.subheader("💰 Average Spending Score per Cluster")
avg_spending = df.groupby('Cluster')['Spending Score (1-100)'].mean().reset_index()

fig, ax = plt.subplots()
sns.barplot(x='Cluster', y='Spending Score (1-100)', data=avg_spending, palette='viridis', ax=ax)
ax.set_title("Average Spending per Cluster")
st.pyplot(fig)


st.subheader("📝 Cluster Insights")
for cluster_id, spending in avg_spending.itertuples(index=False):
    st.write(f"**Cluster {cluster_id}:** Average Spending Score = {spending:.2f}")
