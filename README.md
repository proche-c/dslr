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

### **Installation and Setupc**  

1. Clone the repository:
   ```bash
   git clone <repository_url>
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

3
