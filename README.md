# Decision Tree Regression with scikit-learn

This repository contains a Jupyter Notebook that provides a step-by-step guide to understanding, building, and evaluating decision tree regression models using Python's scikit-learn library. Through practical examples, this notebook demonstrates the process of training a decision tree, visualizing its structure, and assessing its performance on a real-world dataset.

## Overview

### Decision Trees
Decision trees are a popular and intuitive method for both classification and regression tasks. They work by recursively partitioning the data into subsets based on the value of predictor variables, creating a tree-like model of decisions.

### Key Concepts
- **Recursive Binary Splitting**: The method used to partition the data at each node in the tree.
- **Split Criteria**: The predictor variable and its value used to split the data at each node, chosen to minimize the mean squared error (MSE).

## Notebook Content

### Data Visualization and Preparation
The notebook starts by importing necessary libraries and loading the `house_price_by_area` dataset, which contains information on the `LotArea` and corresponding `SalePrice` of properties.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://github.com/Explore-AI/Public-Data/blob/master/house_price_by_area.csv?raw=true")
X = df["LotArea"]
y = df["SalePrice"]

plt.scatter(X, y)
plt.title("House Price vs Area")
plt.xlabel("Lot Area in m²")
plt.ylabel("Sale Price in Rands")
plt.show()
```

### Train-Test Split
The data is split into training and testing sets to evaluate the model's performance on unseen data.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X.values.reshape(-1, 1), y, test_size=0.2, random_state=42)
```

### Building the Decision Tree
A `DecisionTreeRegressor` model is instantiated, trained, and visualized.

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree

regr_tree = DecisionTreeRegressor(max_depth=2, random_state=42)
regr_tree.fit(x_train, y_train)

plt.figure(figsize=(9,9))
_ = plot_tree(regr_tree, feature_names=['LotArea'], filled=True)
```

### Model Evaluation
The model's performance is evaluated using the Mean Squared Error (MSE) metric on the test set.

```python
from sklearn.metrics import mean_squared_error

y_pred = regr_tree.predict(x_test)
MSE = mean_squared_error(y_pred, y_test)
print("Regression decision tree model RMSE is:", np.sqrt(MSE))
```

### Visualizing Model Output
The notebook demonstrates how to visualize the model's predictions as a step function over the input feature range.

```python
x_domain = np.linspace(min(X), max(X), 100)[:, np.newaxis]
y_predictions = regr_tree.predict(x_domain)

plt.figure()
plt.scatter(X, y)
plt.plot(x_domain, y_predictions, color="red", label='predictions')
plt.xlabel("Lot Area in m²")
plt.ylabel("Sale Price in Rands")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```

### Advantages and Disadvantages of Decision Trees
- **Advantages**:
  - Easy to understand and interpret.
  - Handles both categorical and numerical data.
  - Requires minimal data preprocessing.
  - Flexible and extendable using ensemble methods.

- **Disadvantages**:
  - Prone to overfitting.
  - Requires careful parameter tuning.
  - Can be biased if classes are imbalanced.

## Usage
To run this notebook, ensure you have Python and the necessary libraries installed. Clone this repository and open the notebook to follow along with the examples and build your own decision tree models.

```bash
git clone https://github.com/yourusername/DecisionTree-Regression-Notebook.git
cd DecisionTree-Regression-Notebook
jupyter notebook
```

## Conclusion
This notebook serves as a comprehensive guide to decision tree regression, providing practical insights into model building, evaluation, and visualization. It is an excellent resource for data scientists and machine learning practitioners looking to implement decision tree models in their projects.

Contributions and feedback are welcome! Feel free to open issues or submit pull requests to enhance this repository.
