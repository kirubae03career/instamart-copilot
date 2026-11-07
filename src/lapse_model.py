
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lapse prediction dates
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

# Load clean datasets
orders = pd.read_csv(DATA_DIR / "orders.csv")
customers = pd.read_csv(DATA_DIR / "customers.csv")

# Convert date columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Split orders by time
# Features may use only orders before the cutoff.
historical_orders = orders[
    (orders["order_ts"] < CUTOFF)
    & (orders["status"].isin(["completed", "cancelled"]))
].copy()

historical_completed = historical_orders[
    historical_orders["status"] == "completed"
].copy()

# Customers who had at least one completed order before cutoff
cohort_ids = historical_completed["customer_id"].unique()

# Label window: September orders are used only to create the target.
september_completed = orders[
    (orders["order_ts"] >= CUTOFF)
    & (orders["order_ts"] < LABEL_END)
    & (orders["status"] == "completed")
]

september_active_ids = set(september_completed["customer_id"])

# Build the cohort table
cohort = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_active_ids)
).astype(int)

print("=" * 55)
print("CUSTOMER LAPSE COHORT")
print("=" * 55)
print("Cohort customers:", len(cohort))
print("Lapsed customers:", int(cohort["lapsed"].sum()))
print(
    "Lapsed percentage:",
    round(cohort["lapsed"].mean() * 100, 2),
    "%"
)
print("\nTarget label counts:")
print(cohort["lapsed"].value_counts().sort_index())


# --------------------------------------------------
# STEP 2: BUILD HISTORICAL CUSTOMER FEATURES
# --------------------------------------------------

# 1. Recency, frequency and monetary from completed
# orders before the cutoff
rfm_features = historical_completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount_pct=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean")
)

rfm_features["recency"] = (
    CUTOFF - rfm_features["last_order_ts"]
).dt.days

rfm_features = rfm_features.drop(columns=["last_order_ts"])

# 2. Cancellation rate from orders with a final status
status_orders = historical_orders[
    historical_orders["status"].isin(["completed", "cancelled"])
]

status_counts = status_orders.groupby("customer_id").agg(
    total_final_orders=("status", "size"),
    cancelled_orders=(
        "status",
        lambda s: (s == "cancelled").sum()
    )
)

status_counts["cancellation_rate"] = (
    status_counts["cancelled_orders"]
    / status_counts["total_final_orders"]
)

# 3. Most frequently ordered product category

# 3. Most frequently ordered product category
# The category is stored in products.csv, not orders.csv.
products = pd.read_csv(DATA_DIR / "products.csv")

print("Product columns:", products.columns.tolist())

# Attach product categories to historical completed orders
completed_with_category = historical_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_counts = (
    completed_with_category
    .groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_orders")
    .sort_values(
        ["customer_id", "category_orders", "category"],
        ascending=[True, False, True]
    )
)

favorite_category = (
    category_counts
    .drop_duplicates("customer_id")
    .set_index("customer_id")["category"]
    .rename("favorite_category")
)


# 4. Combine features with customer information
feature_data = cohort.set_index("customer_id").join(rfm_features)

feature_data = feature_data.join(
    status_counts[["cancellation_rate"]]
)

feature_data = feature_data.join(favorite_category)

# 5. Customer tenure at the cutoff
feature_data["tenure_days"] = (
    CUTOFF - feature_data["signup_date"]
).dt.days

# 6. Select numeric and categorical features
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "tenure_days",
    "age"
]

categorical_features = [
    "city",
    "gender",
    "favorite_category"
]

feature_columns = numeric_features + categorical_features

X = feature_data[feature_columns].copy()
y = feature_data["lapsed"].copy()

print("\n" + "=" * 55)
print("HISTORICAL FEATURE CHECK")
print("=" * 55)

print("Feature rows:", len(X))
print("Feature columns:", len(X.columns))
print("\nMissing values per feature:")
print(X.isnull().sum().to_string())

print("\nFeature preview:")
print(X.head().to_string())

print("\nTarget counts:")
print(y.value_counts().sort_index().to_string())


# =======================================================
# TRAIN / TEST SPLIT
# =======================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\n" + "=" * 55)
print("TRAIN / TEST SPLIT")
print("=" * 55)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining target counts:")
print(y_train.value_counts().sort_index())

print("\nTesting target counts:")
print(y_test.value_counts().sort_index())


# =======================================================
# PREPROCESSING PIPELINE
# =======================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "tenure_days",
    "age",
]

categorical_features = [
    "city",
    "gender",
    "favorite_category",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
            categorical_features,
        ),
    ]
)

print("\n" + "=" * 55)
print("PREPROCESSING CHECK")
print("=" * 55)

# Fit preprocessing ONLY on training data.
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data without fitting on it.
X_test_processed = preprocessor.transform(X_test)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

print(
    "Scaler mean for frequency:",
    round(
        preprocessor.named_transformers_["numeric"].mean_[1],
        2,
    ),
)

print(
    "Full-cohort mean frequency:",
    round(X["frequency"].mean(), 2),
)


# =======================================================
# LOGISTIC REGRESSION WITH GRID SEARCH
# =======================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(max_iter=1000),
        ),
    ]
)

logistic_parameters = {
    "model__C": [0.01, 0.1, 1, 10, 100],
}

logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

print("\n" + "=" * 55)
print("LOGISTIC REGRESSION GRID SEARCH")
print("=" * 55)

logistic_grid.fit(X_train, y_train)

print("Best parameters:", logistic_grid.best_params_)
print("Best cross-validation F1:", round(logistic_grid.best_score_, 4))

logistic_best_model = logistic_grid.best_estimator_
logistic_predictions = logistic_best_model.predict(X_test)


# =======================================================
# RANDOM FOREST WITH GRID SEARCH
# =======================================================

from sklearn.ensemble import RandomForestClassifier

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)

random_forest_parameters = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20],
}

random_forest_grid = GridSearchCV(
    estimator=random_forest_pipeline,
    param_grid=random_forest_parameters,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

print("\n" + "=" * 55)
print("RANDOM FOREST GRID SEARCH")
print("=" * 55)

random_forest_grid.fit(X_train, y_train)

print("Best parameters:", random_forest_grid.best_params_)
print(
    "Best cross-validation F1:",
    round(random_forest_grid.best_score_, 4),
)

random_forest_best_model = random_forest_grid.best_estimator_
random_forest_predictions = random_forest_best_model.predict(X_test)


# =======================================================
# TEST-SET MODEL EVALUATION
# =======================================================

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

def evaluate_model(model_name, model, X_test, y_test):
    predictions = model.predict(X_test)

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Confusion matrix [[TN, FP], [FN, TP]]:"
    )
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print(
        "Lapsed precision:",
        round(precision_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Lapsed recall:",
        round(recall_score(y_test, predictions, zero_division=0), 4),
    )
    print(
        "Lapsed F1:",
        round(f1_score(y_test, predictions, zero_division=0), 4),
    )

    return f1_score(y_test, predictions, zero_division=0)


logistic_test_f1 = evaluate_model(
    "LOGISTIC REGRESSION — TEST SET",
    logistic_best_model,
    X_test,
    y_test,
)

random_forest_test_f1 = evaluate_model(
    "RANDOM FOREST — TEST SET",
    random_forest_best_model,
    X_test,
    y_test,
)

print("\n" + "=" * 55)
print("TEST-SET MODEL COMPARISON")
print("=" * 55)
print("Logistic Regression F1:", round(logistic_test_f1, 4))
print("Random Forest F1:", round(random_forest_test_f1, 4))

if max(logistic_test_f1, random_forest_test_f1) >= 0.55:
    print("Assignment F1 target (0.55): ACHIEVED")
else:
    print("Assignment F1 target (0.55): NOT YET ACHIEVED")


# =======================================================
# OVERSAMPLING — TRAINING DATA ONLY
# =======================================================

from sklearn.utils import resample
from sklearn.base import clone

print("\n" + "=" * 55)
print("OVERSAMPLING TRAINING DATA")
print("=" * 55)

# Combine training features and labels.
training_data = X_train.copy()
training_data["_lapsed_target"] = y_train

# Separate the two classes.
majority_class = training_data[
    training_data["_lapsed_target"] == 0
]
minority_class = training_data[
    training_data["_lapsed_target"] == 1
]

# Oversample lapsed customers with replacement.
minority_oversampled = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),
    random_state=42,
)

# Combine and shuffle the balanced training set.
balanced_training = pd.concat(
    [majority_class, minority_oversampled]
).sample(frac=1, random_state=42)

X_train_balanced = balanced_training.drop(
    columns="_lapsed_target"
)
y_train_balanced = balanced_training["_lapsed_target"]

print("Balanced training rows:", len(X_train_balanced))
print("\nBalanced training class counts:")
print(y_train_balanced.value_counts().sort_index())

# Clone the best Logistic Regression pipeline.
balanced_logistic_model = clone(logistic_best_model)

# Fit the entire pipeline only on balanced training data.
balanced_logistic_model.fit(
    X_train_balanced,
    y_train_balanced,
)

# Evaluate on the original, unchanged test data.
balanced_predictions = balanced_logistic_model.predict(X_test)

print("\nBalanced Logistic Regression — TEST SET")
print(
    "Precision:",
    round(
        precision_score(
            y_test, balanced_predictions, zero_division=0
        ),
        4,
    ),
)
print(
    "Recall:",
    round(
        recall_score(
            y_test, balanced_predictions, zero_division=0
        ),
        4,
    ),
)
print(
    "F1:",
    round(
        f1_score(
            y_test, balanced_predictions, zero_division=0
        ),
        4,
    ),
)
print(
    "Confusion matrix [[TN, FP], [FN, TP]]:"
)
print(confusion_matrix(y_test, balanced_predictions, labels=[0, 1]))


# =======================================================
# FEATURE IMPORTANCE AND MODEL INTERPRETATION
# =======================================================

import numpy as np
import joblib
from pathlib import Path

print("\n" + "=" * 55)
print("TOP FIVE RANDOM FOREST FEATURE IMPORTANCES")
print("=" * 55)

rf_preprocessor = random_forest_best_model.named_steps["preprocessor"]
rf_model = random_forest_best_model.named_steps["model"]

feature_names = rf_preprocessor.get_feature_names_out()
importances = rf_model.feature_importances_

importance_table = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
}).sort_values("importance", ascending=False)

print(importance_table.head(5).to_string(index=False))

print("\n" + "=" * 55)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 55)

logistic_preprocessor = balanced_logistic_model.named_steps["preprocessor"]
logistic_model = balanced_logistic_model.named_steps["model"]

logistic_feature_names = logistic_preprocessor.get_feature_names_out()

coefficient_table = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": logistic_model.coef_[0],
})

coefficient_table["abs_coefficient"] = (
    coefficient_table["coefficient"].abs()
)

print(
    coefficient_table
    .sort_values("abs_coefficient", ascending=False)
    .head(3)[["feature", "coefficient"]]
    .to_string(index=False)
)

print(
    "\nPositive coefficient: associated with higher predicted lapse odds, "
    "holding other encoded features constant."
)
print(
    "Negative coefficient: associated with lower predicted lapse odds, "
    "holding other encoded features constant."
)

# =======================================================
# SAVE AND RELOAD SELECTED MODEL
# =======================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

selected_model = balanced_logistic_model
model_path = OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(selected_model, model_path)

reloaded_model = joblib.load(model_path)

predictions_before_save = selected_model.predict(X_test)
predictions_after_reload = reloaded_model.predict(X_test)

print("\n" + "=" * 55)
print("MODEL SAVE / RELOAD CHECK")
print("=" * 55)
print("Saved model:", model_path)
print(
    "Predictions match exactly:",
    np.array_equal(
        predictions_before_save,
        predictions_after_reload,
    ),
)

assert np.array_equal(
    predictions_before_save,
    predictions_after_reload,
), "Reloaded model predictions do not match!"

# =======================================================
# CREATE PREMIUM CUSTOMER RETENTION LIST
# =======================================================

# RFM segments were created using the full order history.
# This list is therefore an illustrative post-hoc analysis.
segments_path = OUTPUT_DIR / "customer_segments.csv"
segments = pd.read_csv(segments_path)

premium_ids = set(
    segments.loc[
        segments["segment_name"].eq("Premium"),
        "customer_id",
    ]
)

# Use the selected model's predicted probability of class 1 (lapsed).
test_probabilities = selected_model.predict_proba(X_test)[:, 1]

retention_list = pd.DataFrame({
    "customer_id": X_test.index,
    "lapse_probability": test_probabilities,
    "actually_lapsed": y_test.to_numpy(),
})

retention_list = retention_list[
    retention_list["customer_id"].isin(premium_ids)
].sort_values("lapse_probability", ascending=False)

retention_list = retention_list.head(10)

retention_path = OUTPUT_DIR / "retention_list.csv"
retention_list.to_csv(retention_path, index=False)

print("\n" + "=" * 55)
print("PREMIUM CUSTOMER RETENTION LIST")
print("=" * 55)
print(retention_list.to_string(index=False))
print(
    "Actually lapsed among listed customers:",
    int(retention_list["actually_lapsed"].sum()),
)
print("Retention list saved:", retention_path)
