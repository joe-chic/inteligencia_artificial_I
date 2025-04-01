import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Generate synthetic classification dataset
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Feature importance
importances = rf.feature_importances_

# Plot feature importance
plt.figure(figsize=(6, 4))
sns.barplot(x=np.arange(len(importances)), y=importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance in Random Forest")
plt.show()

# Decision boundary visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6, 5))
cmap_background = ListedColormap(["#FFAAAA", "#AAAAFF"])
cmap_points = ListedColormap(["#FF0000", "#0000FF"])

plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_points, edgecolor="k", label="Training Data")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_points, edgecolor="white", marker="s", label="Test Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Random Forest Decision Boundary (Accuracy: {accuracy:.2f})")
plt.legend()
plt.show()

# Print accuracy and feature importance
print(f"Model Accuracy on Test Set: {accuracy:.2f}")
print(f"Feature Importances: {importances}")
