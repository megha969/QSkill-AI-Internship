# QSkill AI Internship – June 2026

**Domain:** Artificial Intelligence & Machine Learning
**Duration:** 1st June 2026 – 1st July 2026

---

## Task 1: Iris Flower Classification

**Objective:** Classify iris flowers into three species (Setosa, Versicolor, Virginica) based on petal and sepal measurements.

**Dataset:** Iris Dataset from scikit-learn

**Models Used:**
- K-Nearest Neighbors (k=5)
- Logistic Regression
- Decision Tree

**Best Model:** Logistic Regression — Accuracy: 93.3%

**Steps:**
- Loaded and explored dataset visually (histograms, scatter plots, heatmap)
- Split data into 80% train / 20% test
- Applied StandardScaler for feature scaling
- Trained 3 classifiers and compared results
- Evaluated using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix

---

## Task 2: Spam Mail Detector

**Objective:** Classify SMS messages as Spam or Ham (not spam).

**Dataset:** SMS Spam Collection Dataset

**Models Used:**
- Naive Bayes
- Logistic Regression

**Vectorizers Used:**
- Bag of Words
- TF-IDF

**Best Model:** Bag of Words + Logistic Regression — Accuracy: 98.0%, Precision: 100%

**Steps:**
- Loaded and explored dataset (5572 messages, 13.4% spam)
- Preprocessed text (lowercasing, removed stopwords, digits, punctuation)
- Converted text to numeric features using BoW and TF-IDF
- Trained and compared 4 model combinations
- Evaluated using Accuracy, Precision, Recall, F1 Score

---

## Files
| File | Description |
|------|-------------|
| `task1_iris_classification.py` | Iris classification code |
| `task2_spam_detector.py` | Spam detector code |
| `iris_histograms.png` | Feature distribution plots |
| `iris_scatter.png` | Petal scatter plot |
| `iris_confusion_matrix.png` | Iris confusion matrix |
| `iris_model_comparison.png` | Model comparison chart |
| `spam_eda.png` | Spam EDA charts |
| `spam_confusion_matrix.png` | Spam confusion matrix |
| `spam_model_comparison.png` | Spam model comparison chart |

---

## Tools & Libraries
- Python 3
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
