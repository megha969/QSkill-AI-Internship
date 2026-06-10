# =============================================================
# QSkill Internship – Task 2: Spam Mail Detector
# Domain : Artificial Intelligence & Machine Learning
# =============================================================

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── 1. Load Dataset ─────────────────────────────────────────
print("=" * 60)
print("  TASK 2 – SPAM MAIL DETECTOR")
print("=" * 60)

df = pd.read_csv("/home/claude/sms_spam.tsv", sep="\t",
                 header=None, names=["label", "message"])

print(f"\n📦 Dataset shape  : {df.shape}")
print(f"   Label counts   :\n{df['label'].value_counts().to_string()}")
print(f"\n   Sample messages:")
print(df.sample(4, random_state=7)[["label","message"]].to_string(index=False))

# Binary encode labels
df["label_bin"] = (df["label"] == "spam").astype(int)   # spam=1, ham=0

spam_pct = df["label_bin"].mean() * 100
print(f"\n   Spam rate: {spam_pct:.1f}%")

# ── 2. EDA Plots ────────────────────────────────────────────

# 2a. Class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Spam Mail Detector – EDA", fontsize=14, fontweight="bold")

counts = df["label"].value_counts()
axes[0].bar(counts.index, counts.values, color=["#55A868","#C44E52"],
            edgecolor="white", width=0.4)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold", fontsize=11)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count"); axes[0].grid(axis="y", alpha=0.3)

# 2b. Message length distribution
df["msg_len"] = df["message"].str.len()
for lbl, col in zip(["ham","spam"], ["#55A868","#C44E52"]):
    sub = df[df["label"]==lbl]["msg_len"]
    axes[1].hist(sub, bins=40, alpha=0.65, label=lbl, color=col, edgecolor="white")
axes[1].set_title("Message Length Distribution")
axes[1].set_xlabel("Characters"); axes[1].set_ylabel("Count")
axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/spam_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: spam_eda.png")

# ── 3. Text Preprocessing ───────────────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","they","them",
    "their","what","which","who","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","do","does","did",
    "will","would","shall","should","may","might","must","can","could",
    "a","an","the","and","but","if","or","as","at","by","for","with","about",
    "into","to","from","in","out","on","off","then","so","just","not","no",
    "nor","of","up","it","its","s","t","don","doesn","didn","aren","isn",
    "won","couldn","hadn","hasn","haven","let","ll","m","o","re","ve","y"
}

def preprocess(text):
    text = text.lower()                          # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)             # remove digits
    text = re.sub(r"[^a-z\s]", "", text)        # keep letters only
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

df["clean"] = df["message"].apply(preprocess)
print("\n   Preprocessing sample:")
for _, row in df.sample(2, random_state=1).iterrows():
    print(f"   Original : {row['message'][:70]}...")
    print(f"   Cleaned  : {row['clean'][:70]}...")
    print()

# ── 4. Train / Test Split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label_bin"],
    test_size=0.20, random_state=42, stratify=df["label_bin"]
)
print(f"📊 Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 5. Feature Extraction ───────────────────────────────────
vectorizers = {
    "Bag of Words" : CountVectorizer(max_features=5000),
    "TF-IDF"       : TfidfVectorizer(max_features=5000, ngram_range=(1,2)),
}

classifiers = {
    "Naive Bayes"         : MultinomialNB(),
    "Logistic Regression" : LogisticRegression(max_iter=500, random_state=42),
}

# ── 6. Train & Evaluate All Combinations ────────────────────
print("\n" + "─" * 72)
print(f"{'Vectorizer':<15} {'Classifier':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("─" * 72)

all_results = {}
for vname, vec in vectorizers.items():
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    for cname, clf in classifiers.items():
        clf_clone = type(clf)(**clf.get_params())
        clf_clone.fit(Xtr, y_train)
        y_pred = clf_clone.predict(Xte)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        key  = f"{vname} + {cname}"
        all_results[key] = {
            "vec": vec, "clf": clf_clone, "pred": y_pred,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1
        }
        print(f"{vname:<15} {cname:<22} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")

print("─" * 72)

# ── 7. Best Model Deep-Dive ─────────────────────────────────
best_key  = max(all_results, key=lambda k: all_results[k]["f1"])
best      = all_results[best_key]
print(f"\n🏆 Best Combo : {best_key}  (F1 = {best['f1']:.4f})")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, best["pred"], target_names=["Ham","Spam"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, best["pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"],
            ax=ax, linewidths=0.5, annot_kws={"size": 14})
ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("Actual", fontsize=12)
ax.set_title(f"Confusion Matrix\n{best_key}", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/spam_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: spam_confusion_matrix.png")

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
metrics  = ["acc","prec","rec","f1"]
labels   = ["Accuracy","Precision","Recall","F1"]
x        = np.arange(len(metrics))
bar_w    = 0.20
palette  = ["#4C72B0","#55A868","#C44E52","#8172B2"]

for i, (key, res) in enumerate(all_results.items()):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + i*bar_w, vals, bar_w, label=key,
                  color=palette[i], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + bar_w*1.5); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
ax.set_title("Model Comparison – All Vectorizer × Classifier Combos",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/spam_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: spam_model_comparison.png")

# ── 8. Live Prediction Demo ─────────────────────────────────
print("\n🔍 Live Prediction Demo (Best Model):")
samples = [
    "Congratulations! You've won a FREE iPhone. Click here to claim now!!!",
    "Hey, are we still on for lunch tomorrow at 1pm?",
    "URGENT: Your bank account has been suspended. Call 0800-FREE-CASH immediately",
    "Can you please send me the notes from today's lecture?",
]
best_vec = best["vec"]
best_clf = best["clf"]
for msg in samples:
    clean = preprocess(msg)
    feat  = best_vec.transform([clean])
    pred  = best_clf.predict(feat)[0]
    prob  = best_clf.predict_proba(feat)[0]
    label = "🚫 SPAM" if pred == 1 else "✅ HAM "
    conf  = max(prob) * 100
    print(f"   {label}  ({conf:5.1f}% conf)  →  \"{msg[:55]}...\"")

print("\n✅ Task 2 complete!\n")
