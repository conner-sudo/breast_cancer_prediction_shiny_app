library(shiny)
library(bslib)
library(e1071)

# Load the pre-trained SVM model and scaling parameters
package <- readRDS("breast_cancer_svm_model.rds")

# UI
ui <- page_fillable(
  theme = bs_theme(
    bg = "#ffffff",
    fg = "#000000",
    primary = "#0d6efd",
    secondary = "#6c757d",
    success = "#198754",
    danger = "#dc3545"
  ),
  title = "Breast Cancer Risk Assessment",
  
  layout_columns(
    col_widths = c(3, 9),
    
    card(
      card_header("Input Features"),
      card_body(
        h5("Mean Values"),
        numericInput("radius_mean", "Mean radius:", value = 15, min = 5, max = 30, step = 0.01),
        numericInput("texture_mean", "Mean texture:", value = 20, min = 8, max = 40, step = 0.01),
        numericInput("perimeter_mean", "Mean perimeter:", value = 100, min = 40, max = 190, step = 0.01),
        numericInput("area_mean", "Mean area:", value = 700, min = 200, max = 2500, step = 0.1),
        numericInput("smoothness_mean", "Mean smoothness:", value = 0.1, min = 0.05, max = 0.15, step = 0.001),
        numericInput("compactness_mean", "Mean compactness:", value = 0.1, min = 0.02, max = 0.35, step = 0.001),
        numericInput("concavity_mean", "Mean concavity:", value = 0.1, min = 0, max = 0.43, step = 0.001),
        numericInput("concave_points_mean", "Mean concave points:", value = 0.05, min = 0, max = 0.2, step = 0.001),
        numericInput("symmetry_mean", "Mean symmetry:", value = 0.18, min = 0.1, max = 0.3, step = 0.001),
        numericInput("fractal_dimension_mean", "Mean fractal dimension:", value = 0.06, min = 0.05, max = 0.1, step = 0.001),
        
        h5("Standard Error Values"),
        numericInput("radius_se", "Standard error radius:", value = 0.4, min = 0.01, max = 3, step = 0.01),
        numericInput("texture_se", "Standard error texture:", value = 1, min = 0.01, max = 5, step = 0.01),
        numericInput("perimeter_se", "Standard error perimeter:", value = 0.5, min = 0.01, max = 2, step = 0.01),
        numericInput("area_se", "Standard error area:", value = 10, min = 0.01, max = 100, step = 0.1),
        numericInput("smoothness_se", "Standard error smoothness:", value = 0.01, min = 0.001, max = 0.05, step = 0.001),
        numericInput("compactness_se", "Standard error compactness:", value = 0.01, min = 0.001, max = 0.1, step = 0.001),
        numericInput("concavity_se", "Standard error concavity:", value = 0.01, min = 0.001, max = 0.1, step = 0.001),
        numericInput("concave_points_se", "Standard error concave points:", value = 0.005, min = 0.001, max = 0.05, step = 0.001),
        numericInput("symmetry_se", "Standard error symmetry:", value = 0.01, min = 0.001, max = 0.1, step = 0.001),
        numericInput("fractal_dimension_se", "Standard error fractal dimension:", value = 0.001, min = 0.0001, max = 0.01, step = 0.0001),
        
        h5("Worst Values"),
        numericInput("radius_worst", "Worst radius:", value = 8, min = 5, max = 40, step = 0.01),
        numericInput("texture_worst", "Worst texture:", value = 30, min = 8, max = 50, step = 0.01),
        numericInput("perimeter_worst", "Worst perimeter:", value = 50, min = 10, max = 300, step = 0.01),
        numericInput("area_worst", "Worst area:", value = 500, min = 50, max = 5000, step = 0.1),
        numericInput("smoothness_worst", "Worst smoothness:", value = 0.1, min = 0.01, max = 0.2, step = 0.001),
        numericInput("compactness_worst", "Worst compactness:", value = 0.1, min = 0.01, max = 0.3, step = 0.001),
        numericInput("concavity_worst", "Worst concavity:", value = 0.1, min = 0.01, max = 0.3, step = 0.001),
        numericInput("concave_points_worst", "Worst concave points:", value = 0.05, min = 0.01, max = 0.2, step = 0.001),
        numericInput("symmetry_worst", "Worst symmetry:", value = 0.2, min = 0.01, max = 0.4, step = 0.001),
        numericInput("fractal_dimension_worst", "Worst fractal dimension:", value = 0.06, min = 0.01, max = 0.1, step = 0.001),
        
        hr(),
        actionButton("predict_btn", "Predict Risk", class = "btn-primary w-100")
      )
    ),
    
    layout_columns(
      col_widths = c(7, 5, 12),
      
      card(
        card_header("Bulk Data Entry"),
        textAreaInput("bulk_input", "Paste 30 comma-separated values:", 
                      placeholder = "e.g., 17.99, 10.38, 122.8, 1001, 0.1184, ...", rows = 4),
        actionButton("bulk_predict_btn", "Predict from Paste", class = "btn-outline-primary")
      ),
      
      card(
        card_header("Prediction Result"),
        card_body(uiOutput("prediction_result"))
      ),
      
      card(
        card_body(
          tags$small(class = "text-muted",
            "This tool uses a RBF Support Vector Machine (SVM) model trained on ",
            "breast cancer data. Predictions are statistical estimates, not clinical diagnoses. ",
            "If you have health concerns, please consult a medical professional."
          )
        )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Reactive value to store prediction results
  prediction_val <- reactiveVal(NULL)
  
  # Helper function to scale and predict
  run_svm_model <- function(vals) {
    vals_vec <- as.numeric(unlist(vals, use.names = FALSE))
    
    # Extract center and scale from stored parameters
    center_vec <- unlist(package$mean)
    scale_vec <- unlist(package$sd)
    
    expected_count <- length(center_vec)
    
    if (length(vals_vec) != expected_count) {
      stop(sprintf("Input has %d values, but the model expects %d features.", 
                   length(vals_vec), expected_count))
    }
    
    # Create matrix and scale using stored parameters
    input_matrix <- matrix(vals_vec, nrow = 1)
    input_scaled <- scale(input_matrix, center = center_vec, scale = scale_vec)
    
    # Predict
    predict(package$model, input_scaled)
  }
  
  # 1. Manual Prediction Logic
  observeEvent(input$predict_btn, {
    tryCatch({
      vals <- c(
        input$radius_mean, input$texture_mean, input$perimeter_mean, input$area_mean, 
        input$smoothness_mean, input$compactness_mean, input$concavity_mean, 
        input$concave_points_mean, input$symmetry_mean, input$fractal_dimension_mean,
        input$radius_se, input$texture_se, input$perimeter_se, input$area_se, 
        input$smoothness_se, input$compactness_se, input$concavity_se, 
        input$concave_points_se, input$symmetry_se, input$fractal_dimension_se,
        input$radius_worst, input$texture_worst, input$perimeter_worst, 
        input$area_worst, input$smoothness_worst, input$compactness_worst, 
        input$concavity_worst, input$concave_points_worst, input$symmetry_worst, 
        input$fractal_dimension_worst
      )
      prediction_val(run_svm_model(vals))
    }, error = function(e) {
      showNotification(paste("Prediction Error:", e$message), type = "error")
    })
  })
  
  # 2. Bulk Prediction Logic
  observeEvent(input$bulk_predict_btn, {
    req(input$bulk_input)
    tryCatch({
      # Split string by comma, trim whitespace, and convert to numeric
      clean_input <- trimws(unlist(strsplit(input$bulk_input, ",")))
      vals <- as.numeric(clean_input)
      
      if (length(vals) == 30 && !any(is.na(vals))) {
        prediction_val(run_svm_model(vals))
      } else {
        showNotification("Invalid input. Please provide exactly 30 numeric values separated by commas.", type = "error")
      }
    }, error = function(e) {
      showNotification(paste("Error processing input:", e$message), type = "error")
    })
  })
  
  # 3. Render Output
  output$prediction_result <- renderUI({
    res <- prediction_val()
    if (is.null(res)) {
      return(p("Adjust features and click 'Predict' to see result."))
    }
    
    # Check if result is Malignant (handles "M", "1", or 1)
    is_malignant <- res == "M" || res == "1" || res == 1
    
    label <- if (is_malignant) "MALIGNANT" else "BENIGN"
    bg_class <- if (is_malignant) "bg-danger" else "bg-success"
    
    div(
      class = paste("p-4 text-center text-white rounded", bg_class),
      h2(class = "fw-bold", label),
      p(if (is_malignant) "Recommendation: Consult a medical professional immediately." else "Low risk: Regular screening recommended.")
    )
  })
}

shinyApp(ui, server)
