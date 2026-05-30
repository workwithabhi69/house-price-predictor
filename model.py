import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# -----------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------
df = pd.read_csv("House Price Prediction Dataset.csv")
df = df.drop(columns=["Id"], errors="ignore")

# -----------------------------------------------------------
# 2. Correlation Heatmap
# -----------------------------------------------------------
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values)

# Axis labels
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

# Add correlation values
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center", fontsize=8)

ax.set_title("Correlation Heatmap")
fig.colorbar(im)
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# -----------------------------------------------------------
# 3. Define Features and Target
# -----------------------------------------------------------
TARGET = "Price"
FEATURES = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt",
    "Location",
    "Condition",
    "Garage"
]

X = df[FEATURES]
y = df[TARGET]

categorical_features = ["Location", "Condition", "Garage"]
numerical_features = [
    col for col in FEATURES if col not in categorical_features
]

# -----------------------------------------------------------
# 4. Preprocessing
# -----------------------------------------------------------
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# -----------------------------------------------------------
# 5. Train-Test Split
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------------------------
# 6. Model Comparison
# -----------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
}

results = []
rf_pipeline = None
y_pred_rf = None

for model_name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R² Score": round(r2, 4)
    })

    # Save Random Forest model for additional plots
    if model_name == "Random Forest Regressor":
        rf_pipeline = pipeline
        y_pred_rf = y_pred

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="R² Score",
    ascending=False
).reset_index(drop=True)

print("\nModel Performance Comparison\n")
print(results_df.to_string(index=False))

# -----------------------------------------------------------
# 7. Model Comparison Bar Chart
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(results_df))
width = 0.25

ax.bar(x - width, results_df["MAE"], width, label="MAE")
ax.bar(x, results_df["RMSE"], width, label="RMSE")
ax.bar(x + width, results_df["R² Score"], width, label="R² Score")

ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"], rotation=15)
ax.set_title("Model Comparison")
ax.legend()

plt.tight_layout()
plt.savefig("model_comparison_bar_chart.png")
plt.show()

# -----------------------------------------------------------
# 8. Actual vs Predicted Scatter Plot (Random Forest)
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred_rf, alpha=0.6)

# Ideal prediction line
min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())
ax.plot([min_val, max_val], [min_val, max_val], linewidth=2)

ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted Prices (Random Forest)")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# -----------------------------------------------------------
# 9. Feature Importance Chart (Random Forest)
# -----------------------------------------------------------
# Get feature names after preprocessing
feature_names = (
    numerical_features +
    list(
        rf_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
    )
)

# Get importances
importances = (
    rf_pipeline.named_steps["model"].feature_importances_
)

# Sort by importance
indices = np.argsort(importances)[::-1]

# Top 15 features
top_n = min(15, len(importances))
top_indices = indices[:top_n]

top_features = [feature_names[i] for i in top_indices]
top_importances = importances[top_indices]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_features[::-1], top_importances[::-1])
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nGenerated Visualizations:")
print("1. correlation_heatmap.png")
print("2. actual_vs_predicted.png")
print("3. feature_importance.png")
print("4. model_comparison_bar_chart.png")