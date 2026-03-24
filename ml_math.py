import numpy as np

# ─────────────────────────────────────────
# 1. LINEAR REGRESSION
# Math: y = mx + b
# Uses: Least Squares Method to find best fit line
# ─────────────────────────────────────────
def linear_regression(X, y):
    """
    Formula: w = (XᵀX)⁻¹ Xᵀy
    """
    X_b = np.c_[np.ones((len(X), 1)), X]  # add bias term
    weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    print("\n📐 Linear Regression:")
    print(f"  Bias (intercept): {weights[0]:.4f}")
    print(f"  Weight (slope):   {weights[1]:.4f}")
    return weights


# ─────────────────────────────────────────
# 2. LOGISTIC REGRESSION
# Math: sigmoid(z) = 1 / (1 + e^(-z))
# Uses: Gradient Descent to minimize log loss
# ─────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.1, epochs=1000):
    """
    Loss: -[y*log(p) + (1-y)*log(1-p)]
    Update: w = w - lr * Xᵀ(predicted - actual)
    """
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        z = X @ weights
        predicted = sigmoid(z)
        error = predicted - y
        weights -= lr * (X.T @ error) / len(y)
    print("\n📐 Logistic Regression:")
    print(f"  Learned Weights: {weights}")
    return weights


# ─────────────────────────────────────────
# 3. GRADIENT DESCENT
# Math: w = w - lr * gradient
# Core optimization used in almost all ML
# ─────────────────────────────────────────
def gradient_descent(X, y, lr=0.01, epochs=1000):
    """
    MSE Loss: (1/n) * Σ(y - ŷ)²
    Gradient: dL/dw = -(2/n) * Xᵀ(y - Xw)
    """
    w = 0.0
    b = 0.0
    n = len(y)
    for _ in range(epochs):
        y_pred = w * X + b
        dw = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        w -= lr * dw
        b -= lr * db
    print("\n📐 Gradient Descent:")
    print(f"  Final Weight: {w:.4f}")
    print(f"  Final Bias:   {b:.4f}")
    return w, b


# ─────────────────────────────────────────
# 4. EUCLIDEAN DISTANCE (used in KNN)
# Math: d = √Σ(x1 - x2)²
# ─────────────────────────────────────────
def euclidean_distance(point1, point2):
    """
    KNN uses this to find nearest neighbors
    """
    distance = np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
    print("\n📐 Euclidean Distance (KNN Math):")
    print(f"  Distance between {point1} and {point2} = {distance:.4f}")
    return distance


# ─────────────────────────────────────────
# 5. ENTROPY & INFORMATION GAIN (Decision Tree)
# Math: H(S) = -Σ p(x) * log2(p(x))
# ─────────────────────────────────────────
def entropy(labels):
    """
    Measures impurity/uncertainty in a dataset
    """
    n = len(labels)
    classes = np.unique(labels)
    h = 0.0
    for c in classes:
        p = np.sum(labels == c) / n
        h -= p * np.log2(p + 1e-9)
    print("\n📐 Entropy (Decision Tree Math):")
    print(f"  Labels: {labels}")
    print(f"  Entropy: {h:.4f}")
    return h


# ─────────────────────────────────────────
# 6. MEAN SQUARED ERROR (MSE)
# Math: MSE = (1/n) * Σ(actual - predicted)²
# ─────────────────────────────────────────
def mean_squared_error(actual, predicted):
    mse = np.mean((np.array(actual) - np.array(predicted)) ** 2)
    print("\n📐 Mean Squared Error:")
    print(f"  Actual:    {actual}")
    print(f"  Predicted: {predicted}")
    print(f"  MSE:       {mse:.4f}")
    return mse


# ─────────────────────────────────────────
# 7. SOFTMAX (Neural Networks - Multi-class)
# Math: softmax(x) = e^x / Σe^x
# ─────────────────────────────────────────
def softmax(z):
    e_z = np.exp(z - np.max(z))
    result = e_z / e_z.sum()
    print("\n📐 Softmax (Neural Network Math):")
    print(f"  Input:  {z}")
    print(f"  Output: {result}")
    return result


# ─────────────────────────────────────────
# MAIN - RUN ALL
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   Math Behind ML Algorithms")
    print("=" * 50)

    # Linear Regression
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 5, 4, 5], dtype=float)
    linear_regression(X, y)

    # Gradient Descent
    gradient_descent(X, y)

    # Logistic Regression
    X_log = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_log = np.array([0, 0, 1, 1])
    logistic_regression(X_log, y_log)

    # KNN Distance
    euclidean_distance([1, 2], [4, 6])

    # Decision Tree Entropy
    labels = np.array([1, 1, 0, 1, 0, 0, 1])
    entropy(labels)

    # MSE
    mean_squared_error([3, 5, 2, 8], [2.5, 5.1, 2.2, 7.8])

    # Softmax
    softmax(np.array([2.0, 1.0, 0.5]))
```

---

### Step 3: Commit the File

Commit message:
```
Add math behind ML algorithms implementation
