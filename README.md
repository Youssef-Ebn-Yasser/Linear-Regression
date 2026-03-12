# Linear Regression From Scratch (Python)

This project implements **Linear Regression from scratch** using Python and NumPy without using machine learning libraries for the algorithm itself.
The goal is to understand how gradient descent optimizes the weights and bias of a regression model.

## Overview

Linear Regression is a supervised learning algorithm used to model the relationship between input features and a continuous output value.

In this project we:

* Implement Linear Regression manually
* Train the model using Gradient Descent
* Generate synthetic regression data
* Evaluate the model using Mean Squared Error
* Visualize the predictions using a scatter plot

---

## Project Structure

```
project/
│
├── LinearRegression.py
├── main.py
├── README.md
└── regression_result.png
```

---

## Linear Regression Implementation

The `LinearRegression` class implements the algorithm using Gradient Descent.

Key components:

* **Weights**: coefficients of the input features
* **Bias**: intercept of the regression line
* **Learning Rate**: step size used in optimization
* **Iterations**: number of gradient descent updates

Training steps:

1. Initialize weights and bias to zero
2. Compute predictions
3. Calculate gradients
4. Update parameters using gradient descent
5. Repeat for multiple iterations

---

## Training the Model

Synthetic data is generated using `sklearn.datasets.make_regression`.

```
X, y = datasets.make_regression(
    n_samples=100,
    n_features=1,
    noise=20,
    random_state=4
)
```

The dataset is split into training and testing sets using `train_test_split`.

---

## Evaluation

Model performance is evaluated using **Mean Squared Error (MSE)**.

```
def mean_square_error(y_true, y_predict):
    return np.mean((y_true - y_predict) ** 2)
```

Lower MSE means better predictions.

---

## Visualization

The result is visualized using matplotlib:

* Red points → training data
* Blue points → testing data
* Black line → regression prediction

### Regression Result

![Regression Result]<img width="879" height="691" alt="image" src="https://github.com/user-attachments/assets/20f8cca3-32c3-4338-b5e6-fe305ad34c88" />


---

## Requirements

Install dependencies:

```
pip install numpy matplotlib scikit-learn
```

---

## Run the Project

```
python main.py
```

---

## Learning Purpose

This project is useful for understanding:

* Gradient Descent
* Linear Regression mathematics
* Model training workflow
* Machine learning fundamentals

---

## Future Improvements

* Add multiple feature regression
* Implement stochastic gradient descent
* Compare with scikit-learn LinearRegression
* Add R² evaluation metric
