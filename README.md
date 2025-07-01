# Task-6
# 📊 Task 6: K-Nearest Neighbors (KNN) Classification

## 📌 Objective:
Implement and understand the **K-Nearest Neighbors (KNN)** algorithm for classification problems using the Iris dataset.

---

## 📚 Tools & Libraries:
- **Python**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **NumPy**

---

## 📖 Task Description:
- Used the **Iris dataset** for classification.
- Normalized features using `StandardScaler`.
- Split data into **training and testing sets**.
- Applied **K-Nearest Neighbors (KNN)** classification using different K values (1, 3, 5, 7, 9).
- Evaluated model performance with:
  - **Accuracy score**
  - **Confusion Matrix**
- Plotted:
  - **Accuracy vs. K graph**
  - **Decision boundary visualization** using two selected features (`PetalLengthCm` and `PetalWidthCm`).

---

## 📊 Results:
- Achieved maximum accuracy of **95.56% at K=9**.
- Decision boundary plot clearly visualized class regions based on two petal features.
- the output that got from the data
  
K=1 | **Accuracy:** 0.9333
**Confusion Matrix:**
[[15  0  0]
 [ 0 15  0]
 [ 0  3 12]]

K=3 | **Accuracy:** 0.9111
**Confusion Matrix:**
[[15  0  0]
 [ 0 15  0]
 [ 0  4 11]]

K=5 | A**ccuracy:** 0.9111
**Confusion Matrix:**
[[15  0  0]
 [ 0 15  0]
 [ 0  4 11]]

K=7 | **Accuracy:** 0.9333
**Confusion Matrix:**
[[15  0  0]
 [ 0 15  0]
 [ 0  3 12]]

K=9 | **Accuracy:** 0.9556
**Confusion Matrix:**
[[15  0  0]
 [ 0 15  0]
 [ 0  2 13]]


---

## 📌 How to Run:
1. Install required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn

---


