# Load necessary libraries
library(dplyr)
library(e1071)

# 1. Load the Wisconsin Breast Cancer Dataset
# url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
# data <- read.csv(url, header = FALSE)

# 2. Preprocessing
# Define column names as used in the Shiny app
feature_names <- c(
  "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", 
  "compactness_mean", "concavity_mean", "concave.points_mean", "symmetry_mean", "fractal_dimension_mean",
  "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", 
  "compactness_se", "concavity_se", "concave.points_se", "symmetry_se", "fractal_dimension_se",
  "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", 
  "compactness_worst", "concavity_worst", "concave.points_worst", "symmetry_worst", "fractal_dimension_worst"
)
colnames(data) <- c("id", "diagnosis", feature_names)

# Convert diagnosis to factor (M = Malignant, B = Benign)
data$diagnosis <- as.factor(data$diagnosis)

# 3. Scaling
# SVM is highly sensitive to feature scales. 
# We must capture the scaling parameters to use them in the Shiny app.
X <- data[, feature_names]
X_scaled <- scale(X)

# Extract the scaling attributes
scaler_center <- attr(X_scaled, "scaled:center")
scaler_scale <- attr(X_scaled, "scaled:scale")

# 4. Train the SVM Model
# Using a Radial Basis Function (RBF) kernel
final_model <- svm(x = X_scaled, y = data$diagnosis, kernel = "radial", probability = TRUE)

# 5. Save as a Bundle
# This list structure is exactly what the Shiny app expects to find.
bundle <- list(
  model = final_model,
  scaler_center = scaler_center,
  scaler_scale = scaler_scale
)

saveRDS(bundle, "final_model.rds")
message("Success: final_model.rds created with scaling parameters.")