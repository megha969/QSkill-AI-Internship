# =============================================================
# QSkill Internship – Task 1: Iris Flower Classification
# Domain : Artificial Intelligence & Machine Learning
# =============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Colour palette ─────────────────────────────────────────
COLORS   = ["#4C72B0", "#55A868", "#C44E52"]
SPECIES  = ["Setosa", "Versicolor", "Virginica"]

# ── 1. Load Dataset ─────────────────────────────────────────
print("=" * 60)
print("  TASK 1 – IRIS FLOWER CLASSIFICATION")
print("=" * 60)

iris   = load_iris()
X      = pd.DataFrame(iris.data,   columns=iris.feature_names)
y      = pd.Series(iris.target,    name="species")
df     = X.copy()
df["species"] = y
df["species_name"] = df["species"].map({0:"Setosa", 1:"Versicolor", 2:"Virginica"})

print(f"\n📦 Dataset shape : {df.shape}")
print(f"   Classes       : {SPECIES}")
print(f"   Class counts  :\n{df['species_name'].value_counts().to_string()}")
print(f"\n   First 5 rows:\n{df.head().to_string(index=False)}")

# ── 2. Exploratory Data Analysis ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Iris Dataset – Feature Distributions by Species",
             fontsize=15, fontweight="bold", y=1.01)

features = iris.feature_names
for ax, feat in zip(axes.flat, features):
    for cls, col in zip(SPECIES, COLORS):
        vals = df[df["species_name"] == cls][feat]
        ax.hist(vals, bins=15, alpha=0.65, label=cls, color=col, edgecolor="white")
    ax.set_title(feat, fontsize=11, fontweight="bold")
    ax.set_xlabel("cm"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/iris_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: iris_histograms.png")

# Scatter plot – petal length vs petal width (most separable pair)
fig, ax = plt.subplots(figsize=(8, 5))
for cls, col in zip(SPECIES, COLORS):
    sub = df[df["species_name"] == cls]
    ax.scatter(sub["petal length (cm)"], sub["petal width (cm)"],
               label=cls, color=col, alpha=0.75, edgecolors="white", s=60)
ax.set_xlabel("Petal Length (cm)", fontsize=11)
ax.set_ylabel("Petal Width (cm)",  fontsize=11)
ax.set_title("Petal Length vs Petal Width", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/iris_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: iris_scatter.png")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/iris_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: iris_correlation.png")

# ── 3. Train / Test Split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n📊 Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── 4. Feature Scaling ──────────────────────────────────────
scaler  = StandardScaler()
Xtr_sc  = scaler.fit_transform(X_train)
Xte_sc  = scaler.transform(X_test)

# ── 5. Train Multiple Classifiers ───────────────────────────
models = {
    "K-Nearest Neighbors (k=5)" : KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression"        : LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree"              : DecisionTreeClassifier(max_depth=4, random_state=42),
}

results = {}
print("\n" + "─" * 60)
print(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("─" * 60)

for name, model in models.items():
    model.fit(Xtr_sc, y_train)
    y_pred = model.predict(Xte_sc)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    results[name] = {"model": model, "pred": y_pred,
                     "acc": acc, "prec": prec, "rec": rec, "f1": f1}
    print(f"{name:<30} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

print("─" * 60)

# ── 6. Best Model Deep-Dive ─────────────────────────────────
best_name = max(results, key=lambda n: results[n]["f1"])
best      = results[best_name]
print(f"\n🏆 Best Model : {best_name}  (F1 = {best['f1']:.4f})")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, best["pred"], target_names=SPECIES))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best["pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=SPECIES, yticklabels=SPECIES, ax=ax,
            linewidths=0.5, annot_kws={"size": 13})
ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Actual", fontsize=12)
ax.set_title(f"Confusion Matrix – {best_name}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: iris_confusion_matrix.png")

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
metrics  = ["acc", "prec", "rec", "f1"]
labels   = ["Accuracy", "Precision", "Recall", "F1"]
x        = np.arange(len(metrics))
bar_w    = 0.25
bar_cols = ["#4C72B0", "#55A868", "#C44E52"]

for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + i * bar_w, vals, bar_w, label=name,
                  color=bar_cols[i], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + bar_w); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)
ax.set_title("Model Comparison – All Metrics", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("/home/claude/iris_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: iris_model_comparison.png")

print("\n✅ Task 1 complete!\n")
