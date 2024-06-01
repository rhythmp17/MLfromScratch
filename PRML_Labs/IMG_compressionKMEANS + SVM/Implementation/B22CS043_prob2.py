import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Step 1: Load the Iris dataset
iris = datasets.load_iris(as_frame=True)

# Step 2: Select only 'setosa' and 'versicolor' classes
selected_classes = iris.target.isin([0, 1])
X = iris.data[selected_classes]
y = iris.target[selected_classes]

# Step 3: Extract petal length and petal width features
X = X[['petal length (cm)', 'petal width (cm)']]

# Step 4: Normalize the dataset
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Step 5: Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Print the shapes of train and test sets
# print(X_train.shape)

model = LinearSVC()
model.fit(X_train, y_train)

# Step 2: Plot the decision boundary of the model on the training data
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
plt.xlabel('Petal Length (Normalized)')
plt.ylabel('Petal Width (Normalized)')
plt.title('Decision Boundary on Train Data')
plt.show()

# Step 3: Generate a scatterplot of the test data
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm)
plt.xlabel('Petal Length (Normalized)')
plt.ylabel('Petal Width (Normalized)')
plt.title('Scatterplot of Test Data')
plt.show()

# Step 4: Plot the decision boundary of the model on the scatterplot of the test data
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm)
plt.xlabel('Petal Length (Normalized)')
plt.ylabel('Petal Width (Normalized)')
plt.title('Decision Boundary on Test Data')
plt.show()

# Generate synthetic dataset with make_moons
X, y = make_moons(n_samples=500, noise=0.05, random_state=37)

# Add 5% noise (misclassifications)
num_noise = int(0.05 * len(y))
noise_indices = np.random.choice(len(y), num_noise, replace=False)
y[noise_indices] = 1 - y[noise_indices]  # flip the labels

# Plot the synthetic dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset with 5% Noise')
plt.show()

def plot_decision_boundary(model, x_min, x_max, y_min, y_max):
    XX, YY = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.contourf(XX, YY, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Train SVM models with different kernels
linear_svc = SVC(kernel='linear')
poly_svc = SVC(kernel='poly', degree=5)  # Polynomial kernel of degree 3
rbf_svc = SVC(kernel='rbf')  # Radial Basis Function (RBF) kernel

# Fit the models
linear_svc.fit(X, y)
poly_svc.fit(X, y)
rbf_svc.fit(X, y)

# Plot decision boundaries
plt.figure(figsize=(15, 5))

# Linear kernel
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plot_decision_boundary(linear_svc, X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max())
plt.title('Linear Kernel')

# Polynomial kernel
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plot_decision_boundary(poly_svc, X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max())
plt.title('Polynomial Kernel')

# RBF kernel
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plot_decision_boundary(rbf_svc, X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max())
plt.title('RBF Kernel')

plt.tight_layout()
plt.show()

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.1, 0.01, 0.001, 0.0001]}

# Create a GridSearchCV object
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)

# Perform grid search
grid_search.fit(X, y)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best hyperparameters:", best_params)
print("Best cross-validation score:", best_score)

# Train the RBF kernel SVM model with the best hyperparameters
best_gamma = best_params['gamma']
best_C = best_params['C']
best_rbf_svc = SVC(kernel='rbf', gamma=best_gamma, C=best_C)
best_rbf_svc.fit(X, y)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plot_decision_boundary(best_rbf_svc, X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max())
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('RBF Kernel SVM Decision Boundary (Best Hyperparameters)')
plt.show()