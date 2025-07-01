# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
iris_df = pd.read_csv("E:\Iris.csv")  # Replace with your file path if needed
iris_df.drop(columns=['Id'], inplace=True)

# Separate Features and Target
X = iris_df.drop(columns=['Species'])
y = iris_df['Species']

# Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# KNN Classification for Different K Values
k_values = [1, 3, 5, 7, 9]
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)
    print(f"K={k} | Accuracy: {acc:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}\n")

# Plot Accuracy vs. K
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracy_scores, marker='o', color='blue')
plt.title('KNN Accuracy for Different K Values')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# -------------- Decision Boundary Visualization (2 Features) --------------

# Use PetalLengthCm & PetalWidthCm for 2D Plotting
X_2d = iris_df[['PetalLengthCm', 'PetalWidthCm']].values
y_2d = iris_df['Species']

# Normalize
X_2d_scaled = scaler.fit_transform(X_2d)

# Train-Test Split
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d_scaled, y_2d, test_size=0.3, random_state=42, stratify=y_2d)

# Train KNN (K=9)
knn_2d = KNeighborsClassifier(n_neighbors=9)
knn_2d.fit(X_train_2d, y_train_2d)

# Create Mesh Grid
h = .02  # step size in the mesh
x_min, x_max = X_2d_scaled[:, 0].min() - 1, X_2d_scaled[:, 0].max() + 1
y_min, y_max = X_2d_scaled[:, 1].min() - 1, X_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on Mesh
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot Decision Boundaries
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
sns.scatterplot(x=X_2d_scaled[:, 0], y=X_2d_scaled[:, 1],
                hue=y_2d, palette='deep', edgecolor='k')
plt.title('KNN Decision Boundaries (K=9) on 2D Iris Data')
plt.xlabel('PetalLengthCm (scaled)')
plt.ylabel('PetalWidthCm (scaled)')
plt.show()