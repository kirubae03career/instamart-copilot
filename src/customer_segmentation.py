
from pathlib import Path
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the RFM data
rfm = pd.read_csv(
    OUTPUT_DIR / "customer_rfm.csv",
    index_col="customer_id"
)

features = ["recency", "frequency", "monetary"]
X = rfm[features].copy()

# 2. Standardize the features so their scales are comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Test K-Means with different numbers of clusters
results = []

print("=" * 60)
print("K-MEANS CUSTOMER SEGMENTATION")
print("=" * 60)

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    cluster_labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, cluster_labels)

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": score
    })

    print(
        f"K={k} | "
        f"Inertia={model.inertia_:.1f} | "
        f"Silhouette={score:.3f}"
    )

# Save the comparison results
metrics = pd.DataFrame(results)
metrics.to_csv(OUTPUT_DIR / "cluster_metrics.csv", index=False)

# 4. Fit the selected four-cluster model
best_k = 4

final_model = KMeans(
    n_clusters=best_k,
    n_init=10,
    random_state=42
)

rfm["cluster"] = final_model.fit_predict(X_scaled)

# 5. Examine the average RFM values in each cluster

# 5. Examine the average RFM values in each cluster
profile = rfm.groupby("cluster")[features].mean()

print("\nCLUSTER PROFILES")
print(profile.round(2).to_string())

# 6. Assign segment names based on average spending
monetary_order = profile["monetary"].sort_values().index.tolist()

segment_names = ["Budget", "Regular", "High Value", "Premium"]

name_mapping = {
    cluster_id: segment_names[rank]
    for rank, cluster_id in enumerate(monetary_order)
}

rfm["segment_name"] = rfm["cluster"].map(name_mapping)
rfm["segment_id"] = rfm["cluster"]

premium_cluster = profile["monetary"].idxmax()
rfm["is_premium"] = rfm["cluster"] == premium_cluster

# 7. Calculate PCA explained variance
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_scaled)

print("\nPCA EXPLAINED VARIANCE RATIOS")
print(pca.explained_variance_ratio_.round(3))
print("Total:", round(pca.explained_variance_ratio_.sum(), 3))

# 8. Save customer segment results
segment_output = rfm.reset_index()[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

segment_output.to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False
)

print("\nSEGMENT COUNTS")
print(segment_output["segment_name"].value_counts().to_string())

print("\nPREMIUM CUSTOMERS:", int(segment_output["is_premium"].sum()))

print("\nSaved files:")
print(OUTPUT_DIR / "cluster_metrics.csv")
print(OUTPUT_DIR / "customer_segments.csv")