# **Data Science × Logistic Regression**  

## **Introduction**  

This project demonstrates the application of data analysis, visualization, and machine learning techniques to a multi-class classification problem. Using a dataset of
students with numerical features, the objective is to predict the category (house) each student belongs to by implementing a logistic regression classifier.

The project focuses on developing a complete machine learning workflow, including:

- **Data exploration**: analyzing feature distributions, detecting anomalies, and handling missing or inconsistent data.

- **Feature standardization**: normalizing numerical features to prepare them for gradient-based optimization.

- **Visualization**: using histograms, scatter plots, and pair plots to identify correlations and patterns in the data.

- **Logistic regression**: implementing a one-vs-all multi-class classifier trained with gradient descent.

- **Model evaluation and prediction**: generating house predictions for new data and validating model performance.

All computations, including descriptive statistics, standardization, and gradient updates, are implemented manually, avoiding the use of high-level library functions
that would perform these calculations automatically. This approach emphasizes a deep understanding of the underlying algorithms and data handling techniques.   

### **Installation and Setup**  

1. Clone the repository:
   ```bash
   git clone git@github.com:proche-c/dslr.git
   cd dslr
   ```
   
2. Run the installation script to create a virtual environment and install dependencies:
  ```bash
 ./install.sh
``` 

## **Data Analysis: Describe Tool**

This module implements a command-line utility that performs exploratory data analysis on the training dataset. It replicates the **core behavior of the pandas.DataFrame.describe()** function, computing the same fundamental statistical metrics while adhering to the project requirement of implementing all analytical logic manually.

For every numerical feature, the tool computes:

- Count

- Mean

- Standard deviation

- Minimum and maximum values

- Quartiles and percentiles

All parsing, numerical processing, and statistical computations are written from scratch without relying on high-level data analysis utilities. This ensures full transparency over the underlying logic and provides a deeper understanding of how these descriptive statistics are derived.

The Describe tool helps identify potential data quality issues—including missing values, outliers, and skewed distributions—and serves as an essential initial step before proceeding with visualization, standardization, and model training.  

```bash
pipenv run python3 describe.py <path_to_train_dataset>
```

In addition, the tool provides an optional --test mode that compares the output of the custom Describe class with the results produced by pandas.DataFrame.describe().
This feature allows you to validate the correctness and numerical stability of the manual implementation against a trusted reference:

```bash
pipenv run python3 describe.py <path_to_train_dataset> --test
```

## Data Visualization

The project includes a set of visualization tools designed to explore relationships, distributions, and structural patterns within the Hogwarts dataset.
All visualizations are generated using Matplotlib and Seaborn, but every statistical computation (means, variances, correlations, standardization…) is implemented manually, in accordance with project requirements.

These tools provide insight into:

- Feature homogeneity across Hogwarts houses

- Inter-feature correlations

- Feature distributions

- Overall structure of the dataset

- Potential relationships useful for training the logistic regression model

Each visualization module is executed via a standalone script that takes the dataset path as its main argument.  

### Histogram Analysis

This visualization identifies the most homogeneous feature, meaning the feature whose mean value changes the least between Hogwarts houses.

**How it works**

1. Computes the mean of each feature per house using the custom Describe class.

2. Calculates the variance between group means (inter-group variance).

3. Selects the feature with the lowest variance (most homogeneous).

4. Displays:

   - A message with variances of all features.

   - Two histograms of the selected feature:

      - A Matplotlib overlay with custom colors per house

      - A Seaborn histogram with hue separation

**Usage**  

```bash
pipenv run python3 histogram.py <path_to_train_dataset>
```

### Scatter Plot Analysis

The scatter plot tool identifies the two features with the strongest Pearson correlation (positive or negative), computed manually.

**How it works**

1. Selects all numeric columns.

2. Computes Pearson correlation manually for every unique feature pair.

3. Identifies the pair with the strongest absolute correlation.

4. Generates two scatter plots:

   - Raw values

   - Manually standardized values (z-score)

**Usage**  

```bash
pipenv run python3 scatter_plot.py <path_to_train_dataset>
```

### Pair Plot Analysis

This tool provides a broad overview of numeric features by combining results from homogeneity and correlation analysis, and generating a Seaborn pairplot.

**How it works**

1. Computes:

   - Variance between house means (homogeneity)

   - Pearson correlations between all numeric feature pairs

2. Prints:

   - Strongly correlated pairs (|pearson| ≥ 0.8)


   - Features with very low variance between means (≤ 50)

3. Creates a pairplot of all numeric features, colored by house.

**Usage**  

```bash
pipenv run python3 pair_plot.py <path_to_train_dataset>
```

## Logistic Regression

This module implements a multiclass logistic regression classifier from scratch, without relying on machine learning libraries.
Its purpose is to predict the Hogwarts house of each student based on three numerical features.

The system is composed of two main scripts:

- logreg_train.py — trains a one-vs-all multi-class logistic regression model.

- logreg_predict.py — uses the trained weights to predict houses for new data.  

This module is fully self-contained and numerically stable (log-loss, sigmoid, overflow-safe exponentiation, and detailed error handling).  

### Features

✔️ **Training**

- Standardizes all features (mean & std computed from the training set).

- Trains four binary logistic models, one per house (OvA strategy).

- Implements:

   - gradient descent

   - log-loss with eps stabilization

   - convergence detection

   - step-by-step cost tracking for plotting

   - early stopping using min_step_size

✔️ **Prediction**

- Loads the weights from weights.json.

- Validates JSON structure and numerical validity.

- Converts input data to numeric, drops invalid rows, standardizes features.

- Computes predictions safely using overflow-protected sigmoid.

- Selects the class with maximum probability.

✔️ **Evaluation** (optional)

- When --test is used, the prediction script compares your manual implementation with:

- **scikit-learn** LogisticRegression

- automatic scaling

- accuracy evaluation per house

- discrepancy detection between both models

**Outputs are saved as**:

- weights.json - parameters from training

- houses.csv — predictions

- compare.csv — comparison with sklearn model

### **Training the Model**  

Run the training script:

```bash
pipenv run python3 logreg_train.py <path_to_train_dataset>
```

| Parameter                  | Default | Purpose                  |
| -------------------------- | ------- | ------------------------ |
| `--max_steps` / `-ms`      | 15000   | Maximum GD iterations    |
| `--min_step_size` / `-mss` | 0.00005 | Early stopping threshold |
| `--lr`                     | 0.01    | Learning rate            |

Example:  

```bash
pipenv run python logreg_train.py <path_to_train_dataset> --lr 0.005 -ms 20000
```



