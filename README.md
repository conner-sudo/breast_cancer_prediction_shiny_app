# Breast Cancer Risk Assessment Shiny App

[![Live App](https://img.shields.io/badge/Live-Shiny_App-blue?style=for-the-badge&logo=R)](https://connerspear.shinyapps.io/breast_cancer_app/)
![thumbnail]([thumbnail_breast_cancer_app.png](https://github.com/conner-sudo/breast_cancer_prediction_shiny_app/tree/main/images))

## Overview
This repository contains the code and documentation for a production-ready **R Shiny application** designed to predict breast cancer diagnoses (Malignant or Benign) based on cellular features. 

The core predictive engine achieves an exceptionally high **accuracy of 99.1%**, utilizing a highly tuned Support Vector Machine (SVM) that acts as a reliable clinical decision-support tool.

## 🧠 Statistical & Machine Learning Architecture

This project moves beyond standard high-accuracy models to properly address the nuances and real-world consequences of medical diagnostic data.

### 1. The Main Model: Cost-Sensitive Support Vector Machine (SVM)
In medical diagnostics, the consequences of misclassification are rarely symmetric. While predicting a false positive causes unnecessary stress, predicting a false negative (missing a malignant tumor) can be life-threatening. 

To statistically account for this, the app employs a **Cost-Sensitive SVM Classifier**. 
* **Cost-Sensitive Learning:** Explicitly penalizes false negatives during model training, heavily prioritizing the correct identification of malignant cases (maximizing recall).
* **Robust Validation:** Hyperparameters were tuned via Grid Search, and performance was validated using K-Fold Cross Validation ($k=10$) built from scratch to prevent overfitting and ensure out-of-sample generalization.
* **Feature Scaling:** Because SVMs are highly sensitive to the scale of input features due to distance-based margin maximization, all 30 numeric inputs are meticulously centered and scaled (mean of 0, variance of 1) before inference.

## 📊 Features & Predictors
The model processes 30 numeric variables computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features capture the **Mean**, **Standard Error**, and **Worst (largest)** values for the following cell nuclei characteristics:
* **Structural:** Radius, Perimeter, Area
* **Surface & Texture:** Smoothness, Texture
* **Shape & Irregularity:** Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension

## 🚀 Why this project matters
For employers reviewing this repository, this project demonstrates:
1. **Domain-Aware Modeling:** Identifying that standard accuracy metrics fall short in medical contexts and applying cost-sensitive machine learning techniques to actively minimize false negatives.
2. **Rigorous Validation & Pipeline Engineering:** Building data standardization pipelines and iterating through K-Fold cross-validation manually to guarantee the model's reliability in production.
3. **End-to-End Deployment:** Taking an idea from raw Kaggle data processing and model tuning in R (Jupyter Notebook), to serializing the model, building a clean UI with `bslib`, and deploying it as a live, interactive web application via Shiny that supports both single-patient inputs and bulk data processing.
