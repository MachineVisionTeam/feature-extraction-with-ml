# Feature Extraction and Machine Learning for Nuclei Classification

## Project Overview
This project evaluates the effectiveness of hand-crafted features extracted from histopathological images in classifying different nuclei types. Using the PanNuke dataset, which includes labeled H&E-stained tissue images, the project performs feature extraction, selection, and model evaluation. Our goal is to classify nuclei into categories such as neoplastic, inflammatory, and epithelial cells by optimizing feature selection and assessing performance across multiple machine learning models.

## Key Components
- **Feature Extraction**: Extracts 46 hand-crafted features (e.g., morphometric, intensity, gradient, Haralick) from labeled nuclei images.
- **Feature Selection**: Optimizes feature subset selection using Bayesian optimization with methods like ANOVA, RFE, and LASSO.
- **Machine Learning Models**: Includes Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, and a weighted ensemble model to classify nuclei types.

---

## Folder Structure

### `src/`
Source code for feature extraction and machine learning.

### `results/`
Contains evaluation metrics, tables, and plots generated from feature extraction and machine learning experiments.

---

## Features Extracted

The feature extraction process focuses on four main types of features:

- **Morphometric Features**: Area, Perimeter, Eccentricity, Circularity, Major Axis Length, Minor Axis Length, Solidity, etc.
- **Intensity Features**: Min, Max, Mean, Median, Standard Deviation, Entropy, Energy, Skewness, Kurtosis, etc.
- **Gradient Features**: Mean, Standard Deviation, Skewness, Kurtosis, and Canny Mean of the gradient.
- **Haralick Features**: Texture descriptors like ASM, Contrast, Correlation, IDM, Sum Average, Sum Entropy, and others.

Each feature type provides valuable insights into cell morphology, intensity patterns, and texture, contributing to effective classification.

---

## Machine Learning Models

The following eight machine learning models are implemented and evaluated:

- **Logistic Regression (LR)**
- **Decision Tree (DT)**
- **Random Forest (RF)**
- **XGBoost (XGB)**
- **LightGBM (LGBM)**
- **Support Vector Machine (SVM)**
- **Gradient Boosting (GB)**
- **Weighted Ensemble (WE)**: Combines predictions from individual models based on their F1-scores.

Each model is trained and optimized to classify nuclei types based on the extracted features. The ensemble model improves classification accuracy by assigning weights to models with higher performance.

---

## Evaluation Metrics

The models are evaluated on the following metrics:

- **Precision**: Proportion of true positive predictions among all positive predictions.
- **Recall**: Proportion of true positive predictions among all actual positives.
- **F1-score**: Harmonic mean of precision and recall, useful for imbalanced data.
- **Accuracy**: Overall correctness of the model’s predictions.

Evaluation results are stored in the `results/` directory, with plots and tables illustrating each model’s performance across different tissue types.

---

## Results

The results indicate that **ensemble models** consistently outperform individual models, achieving higher precision, recall, and F1-scores across tissue types. Refer to `results/` for detailed analysis and comparative performance.

---

## Scripts Overview

- **Feature Extraction (`feature_extraction.py`)**: Extracts relevant features from histopathological images.
- **Feature Selection (`feature_selection.py`)**: Optimizes the feature set.
- **Model Training (`model_training.py`)**: Trains machine learning models on the selected features.
- **Evaluation (`evaluation.py`)**: Computes metrics to evaluate model performance.

---

## Results Directory (`results/`)

Contains evaluation metrics, tables, and plots generated from feature extraction and machine learning experiments.

---

## Acknowledgments

This research was partially supported by the National Science Foundation under Grant No. 2409704 and Grant No. 2409705.
