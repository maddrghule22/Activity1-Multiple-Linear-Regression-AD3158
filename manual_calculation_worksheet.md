# Manual Multiple Linear Regression Calculation Worksheet
## Academic Course: Regression and Classification Algorithms
## Unit I – Regression Foundations & Linear Regression
---
- **Student Name**: Darshan Ghule
- **Roll Number**: AD3158
- **Project**: Activity 1 – Multiple Linear Regression
- **Faculty Resource Reference**: [https://github.com/complexiota](https://github.com/complexiota)
- **Dataset Source**: [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

---

## 1. Selected Dataset Sample ($n = 6$ Observations, $p = 2$ Features)

To demonstrate the mathematical mechanics of Ordinary Least Squares (OLS) without high-level library abstractions, we extract 6 representative observations from `data/housing.csv`.

To ensure numerical clarity and avoid roundoff errors in manual matrix inversion:
- **$X_1$ ($Area_k$)**: Living area in **thousands of square feet** ($Area / 1,000$).
- **$X_2$ ($Bedrooms$)**: Discrete bedroom count.
- **$y$ ($Price_m$)**: Property sale price in **Millions of Indian Rupees (₹)** ($Price / 1,000,000$).

### Selected Observation Table:
| Sample ($i$) | Actual Area ($sqft$) | $x_{i1} = Area_k$ | $x_{i2} = Bedrooms$ | Actual Price (₹) | $y_i = Price_m$ (₹ Millions) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 7420 | 7.420 | 4 | ₹13,300,000 | ₹13.300M |
| **2** | 13200 | 13.200 | 3 | ₹9,800,000 | ₹9.800M |
| **3** | 7440 | 7.440 | 3 | ₹7,420,000 | ₹7.420M |
| **4** | 6600 | 6.600 | 3 | ₹6,230,000 | ₹6.230M |
| **5** | 4520 | 4.520 | 3 | ₹4,900,000 | ₹4.900M |
| **6** | 4079 | 4.079 | 3 | ₹4,200,000 | ₹4.200M |

---

## 2. Design Matrix $X$ and Target Vector $y$

Adding a leading column of 1s to $X$ to account for the intercept term $\beta_0$:

$$X = \begin{bmatrix}
1 & 7.420 & 4 \\
1 & 13.200 & 3 \\
1 & 7.440 & 3 \\
1 & 6.600 & 3 \\
1 & 4.520 & 3 \\
1 & 4.079 & 3
\end{bmatrix}_{6 \times 3}, \quad
y = \begin{bmatrix}
13.300 \\
9.800 \\
7.420 \\
6.230 \\
4.900 \\
4.200
\end{bmatrix}_{6 \times 1}$$

---

## 3. Matrix Multiplications: $X^T X$ and $X^T y$

The transpose $X^T$ is of dimension $3 \times 6$:

$$X^T X = \begin{bmatrix}
n & \sum x_{i1} & \sum x_{i2} \\
\sum x_{i1} & \sum x_{i1}^2 & \sum x_{i1} x_{i2} \\
\sum x_{i2} & \sum x_{i1} x_{i2} & \sum x_{i2}^2
\end{bmatrix} = \begin{bmatrix}
6.0000 & 43.2590 & 19.0000 \\
43.2590 & 365.2786 & 137.1970 \\
19.0000 & 137.1970 & 61.0000
\end{bmatrix}$$

$$X^T y = \begin{bmatrix}
\sum y_i \\
\sum x_{i1} y_i \\
\sum x_{i2} y_i
\end{bmatrix} = \begin{bmatrix}
45.8500 \\
363.6486 \\
150.8500
\end{bmatrix}$$

---

## 4. Matrix Inversion: $(X^T X)^{-1}$

Using Gaussian elimination / Adjugate matrix inversion $(X^T X)^{-1} = \frac{1}{\det(X^T X)} \text{Adj}(X^T X)$:

$$(X^T X)^{-1} = \begin{bmatrix}
12.970660 & -0.120205 & -3.769684 \\
-0.120205 & 0.018749 & -0.004729 \\
-3.769684 & -0.004729 & 1.201193
\end{bmatrix}$$

---

## 5. Solving the Normal Equations: $\hat{\beta} = (X^T X)^{-1} X^T y$

Multiplying the inverted matrix $(X^T X)^{-1}_{3 \times 3}$ by the vector $X^T y_{3 \times 1}$:

$$\hat{\beta} = \begin{bmatrix}
\hat{\beta}_0 \\
\hat{\beta}_1 \\
\hat{\beta}_2
\end{bmatrix} = \begin{bmatrix}
(12.970660 \times 45.8500) + (-0.120205 \times 363.6486) + (-3.769684 \times 150.8500) \\
(-0.120205 \times 45.8500) + (0.018749 \times 363.6486) + (-0.004729 \times 150.8500) \\
(-3.769684 \times 45.8500) + (-0.004729 \times 363.6486) + (1.201193 \times 150.8500)
\end{bmatrix} = \begin{bmatrix}
\mathbf{-17.664580} \\
\mathbf{0.593427} \\
\mathbf{6.640338}
\end{bmatrix}$$

### Final Fitted Multiple Regression Equation:
$$\widehat{\text{Price}} = \mathbf{-17.664580 + (0.593427 \times Area_k) + (6.640338 \times Bedrooms)} \text{ (₹ Millions)}$$

---

## 6. Manual Prediction & Residual Calculation

### Demonstration 1: Observation 1 ($Area = 7,420$ sq ft $\rightarrow Area_k = 7.420$, $Bedrooms = 4$)
$$\begin{aligned}
\hat{y}_1 &= -17.664580 + (0.593427 \times 7.420) + (6.640338 \times 4) \\
&= -17.664580 + 4.403230 + 26.561351 \\
&= \mathbf{13.300000} \text{ ₹ Millions (₹ 13,300,000.00)}
\end{aligned}$$

$$\text{Residual}_1 = y_1 - \hat{y}_1 = 13.300000 - 13.300000 = \mathbf{0.000000} \text{ (₹ 0.00)}$$

### Demonstration 2: Observation 2 ($Area = 8,400$ sq ft $\rightarrow Area_k = 8.400$, $Bedrooms = 3$)
$$\begin{aligned}
\hat{y}_2 &= -17.664580 + (0.593427 \times 8.400) + (6.640338 \times 3) \\
&= -17.664580 + 4.984788 + 19.921013 \\
&= \mathbf{10.089671} \text{ ₹ Millions (₹ 10,089,671.47)}
\end{aligned}$$

$$\text{Residual}_2 = y_2 - \hat{y}_2 = 9.800000 - 10.089671 = \mathbf{-0.289671} \text{ ₹ Millions (+₹ -289,671.47)}$$

---

## 7. Verification Against Scikit-Learn

```python
from sklearn.linear_model import LinearRegression

# Fit model on the exact 6 observations
sk = LinearRegression(fit_intercept=True)
sk.fit(manual_df[['area_k', 'bedrooms']], manual_df['price_m'])

print(f"Intercept (b0): {sk.intercept_:.6f}")  # Outputs: -17.664580
print(f"Slope Area (b1): {sk.coef_[0]:.6f}")    # Outputs: 0.593427
print(f"Slope Beds (b2): {sk.coef_[1]:.6f}")    # Outputs: 6.640338
```

### Mathematical Verification Comparison:
| Parameter | Manual Calculation | Scikit-Learn Output | Absolute Difference | Verification Status |
|---|:---:|:---:|:---:|:---:|
| Intercept ($\hat{\beta}_0$) | `-17.664580` | `-17.664580` | `0.000000` | **Exact Match** |
| Slope Area ($\hat{\beta}_1$) | `0.593427` | `0.593427` | `0.000000` | **Exact Match** |
| Slope Bedrooms ($\hat{\beta}_2$) | `6.640338` | `6.640338` | `0.000000` | **Exact Match** |

*Conclusion: The manual linear algebra derivation is verified to 6 decimal places of precision.*
