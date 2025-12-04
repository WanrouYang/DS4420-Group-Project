# Using Collaborative Filtering to recommend therapy treatment plan for mental health

# Sample severity scores from CNN (0-10)
severity_score_cnn <- runif(5, min = 0, max = 10)
severity_score_cnn
# severity_score_cnn <- c(10, 6.38, 7.61, 0,4,5.45,9.88)

# Load dataset
treatment_data <- read.csv("mental_health_diagnosis_treatment_.csv")

# Data Cleaning (Keep the columns that we need)
cleaned_df <- treatment_data[, c("Patient.ID",
                                 "Symptom.Severity..1.10.",
                                 "Therapy.Type",
                                 "Outcome")]

colnames(cleaned_df) <- c("patient","severity","therapy","outcome")

# Convert outcomes to ratings (Improved=1, No Change=0, Deteriorated=-1)
cleaned_df$rating <- ifelse(cleaned_df$outcome == "Improved", 1,
                            ifelse(cleaned_df$outcome == "No Change", 0, -1))


# Create the therapy matrix (user-item matrix: a matrix of patient by therapy with their outcome rating)
therapy_table <- xtabs(rating ~ patient + therapy, data = cleaned_df)
therapy_matrix <- as.matrix(therapy_table)

# Set up patient feature for later similarity calculation

# Extract the severity scores from patients
patient_severity_df <- treatment_data[!duplicated(treatment_data$Patient.ID),
                                      c("Patient.ID","Symptom.Severity..1.10.")]
colnames(patient_severity_df) <- c("patient","severity")
rownames(patient_severity_df) <- patient_severity_df$patient
patient_severity_df <- patient_severity_df[rownames(therapy_matrix), , drop=FALSE]
severity_vector <- patient_severity_df$severity

# Get the other numeric features
feature_df <- treatment_data[!duplicated(treatment_data$Patient.ID),
                             c("Patient.ID",
                               "Age",
                               "Mood.Score..1.10.",
                               "Sleep.Quality..1.10.",
                               "Physical.Activity..hrs.week.",
                               "Stress.Level..1.10.",
                               "Treatment.Progress..1.10.",
                               "Adherence.to.Treatment....")]

rownames(feature_df) <- feature_df$Patient.ID
feature_df <- feature_df[rownames(therapy_matrix), , drop=FALSE]
patient_features <- feature_df[, -1] 

# Scale the feature value
patient_features_scaled <- scale(patient_features)


# Cosine similarity
library(lsa)

# Similarity (patient_features)
sim_feature <- cosine(t(patient_features_scaled))
diag(sim_feature) <- NA  

# Similarity (therapy_matrix)
therapy_scaled <- scale(therapy_matrix, scale = FALSE)
sim_therapy <- cosine(t(therapy_scaled))
diag(sim_therapy) <- NA


# Combination of feature and therapy
w_feature <- 0.6
w_therapy <- 0.4
sim_hybrid <- w_feature * sim_feature + w_therapy * sim_therapy

# Min–max Scale
sim_hybrid_scaled <- apply(sim_hybrid, 2, function(x){
  (x - min(x, na.rm=TRUE)) / (max(x, na.rm=TRUE) - min(x, na.rm=TRUE))
})


# Create the recommend_therapy function
recommend_therapy <- function(severity_score_cnn,
                              severity_vector,
                              therapy_matrix,
                              sim_hybrid_scaled,
                              k = 5) {
  
  # If severity score is 0, which means no need for treatment
  if (severity_score_cnn == 0) {
    return(list(
      severity_score_cnn = severity_score_cnn,
      best_therapy = "No treatment needed"
    ))
  }
  
  # Find the closed patient based on severity
  distance <- abs(severity_vector - severity_score_cnn)
  anchor_idx <- which.min(distance)
  
  sims <- sim_hybrid_scaled[, anchor_idx]
  sims_clean <- sims[!is.na(sims)]
  
  # It's for if there is no similar patients, return "No similar patients found"
  if (length(sims_clean) == 0) {
    return(list(
      severity_score_cnn = severity_score_cnn,
      best_therapy = "No similar patients found"
    ))
  }
  
  # Find the most similar patients k
  k <- min(k, length(sims_clean))
  top_idx <- order(sims_clean, decreasing=TRUE)[1:k]
  valid_positions <- which(!is.na(sims))
  idx_neighbor <- valid_positions[top_idx]
  
  # Take the therapy results from patients
  sub_matrix <- therapy_matrix[idx_neighbor, , drop=FALSE]
  w <- sims[idx_neighbor]
  
  # Weighted average to predictions
  # Only use the "Improved" Outcome Rating
  pred_ratings <- apply(sub_matrix, 2, function(col){
    improved_only <- ifelse(col == 1, 1, NA)
    if (all(is.na(improved_only))) return(NA)
    sum((col == 1) * w, na.rm=TRUE) / sum(w, na.rm=TRUE)
  })

  # If there is nothing found, get back to the most successful therapy
  if (all(is.na(pred_ratings))) {
    num_improved <- tapply(cleaned_df$rating == 1, cleaned_df$therapy, sum)
    best_therapy <- names(num_improved)[which.max(num_improved)]
  } else {
    best_therapy <- names(pred_ratings)[which.max(pred_ratings)]
  }
  
  list(
    severity_score_cnn = severity_score_cnn,
    idx_neighbor = idx_neighbor,
    pred_ratings = pred_ratings,
    best_therapy = best_therapy
  )
}


# Try to run this recommendations system
output <- lapply(severity_score_cnn, function(sv){
  recommend_therapy(sv,
                    severity_vector,
                    therapy_matrix,
                    sim_hybrid_scaled,
                    k = 8)
})

# Print out the results
for (i in seq_along(output)) {
  cat("User", i,
      "- Severity:", output[[i]]$severity_score_cnn,
      "-> Therapy:", output[[i]]$best_therapy, "\n")
}


# Show improved outcome count per therapy
#cat("\nNumber of Improved Outcomes per Therapy:\n")
#print(tapply(cleaned_df$rating == 1, cleaned_df$therapy, sum))

# Create the visualization for application
library(dplyr)
library(ggplot2)
library(plotly)

# Set up for plots
outcome_summary <- cleaned_df %>%
  group_by(therapy, outcome) %>%
  summarise(count = n(), .group = "drop_last") %>%
  group_by(therapy) %>%
  mutate(total = sum(count),
         proportion = count / total) %>%
  ungroup()

# Static ggplot
plot <- ggplot(outcome_summary, aes(x = therapy, y = proportion, fill = outcome, text = paste0(
  "Therapy: ", therapy, "\nOutcome: ", outcome, "\nCount: ", count, "\nRate: ", round(proportion, 2)
))) +
  geom_bar(stat = "identity", position = "fill") +
  labs(title = "Outcome Distribution by Therapy Type (Interactive Plot)",
       x = "Therapy Type", y = "Proportion of Outcomes") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))

# Convert to interactive plot
interactive_plot <- ggplotly(plot, tooltip = "text")
interactive_plot

# Using Shiny to build up the application/webpages
library(shiny)
library(grid)
library(png)

landing_page <- fluidPage(
  h1("Therapy Recommendation System"),
  h2("Background"),
  
  p("Mental-health diagnosis today relies heavily on subjective methods 
     such as interviews, self-reports, and clinician observations. Many mental-health conditions—such as 
     depression, anxiety, and mood disorders—also manifest through subtle changes in facial expression, 
     behavior, and emotional patterns. Machine learning offers a way to analyze these signals objectively."),
  
  p("Our project explores a combined approach using:"),
  tags$ul(
    tags$li("Convolutional Neural Networks (CNNs) to recognize facial-emotion cues"),
    tags$li("Collaborative Filtering (CF) to recommend personalized therapy options based on patterns in patient outcomes")
  ),
  
  p("This dual-method framework reflects how emerging AI systems can support clinicians by providing 
     more consistent, data-driven insights."),
  
  hr(),
  
  h2("Project Workflow (What We Did)"),
  
  h3("1. Data Collection & Preparation"),
  
  p("CNN Dataset: We used a Kaggle facial-expression dataset containing seven emotion categories 
     (angry, disgust, fear, happy, neutral, sad, surprise). Images were converted to grayscale, padded, resized 
     to 128×128, normalized, and reshaped into tensors for CNN processing."),
  
  p("CF Dataset: We used a Kaggle mental-health treatment dataset containing patient symptoms, severity scores, 
     therapy types, and treatment outcomes. Data was cleaned, encoded, and transformed into:"),
  tags$ul(
    tags$li("A patient feature matrix (mood, stress, sleep, etc.)"),
    tags$li("A therapy outcome matrix (improved / no change / deteriorated)")
  ),
  
  h3("2. CNN Emotion-Recognition Model"),
  
  p("We built a custom TensorFlow/Keras CNN with two convolution + max-pooling blocks, dense layers, and a 
     7-class softmax output. We implemented data augmentation (rotation, shift, zoom, flip) to reduce overfitting."),
  
  p("The model trained for 30–50 epochs using Adam optimizer and achieved roughly:
     • ~32% training accuracy
     • ~26% validation accuracy
     • Above the 14% random-guess baseline"),
  
  p("Some overfitting was observed due to dataset limitations. The CNN is not diagnostic but provides a signal 
     that can contribute to a larger decision-support pipeline."),
  
  h3("3. Collaborative Filtering Therapy Recommendation Model"),
  
  p("Because CNN severity estimation was not stable enough for direct use, we simulated realistic severity scores 
     in R to test the CF pipeline."),
  
  tags$ul(
    tags$li("Patient Feature Similarity (Cosine Similarity) on standardized mood/stress/sleep variables"),
    tags$li("Therapy-Outcome Similarity based on improvement patterns"),
    tags$li("Hybrid Similarity Score = 0.6 × feature similarity + 0.4 × therapy outcome similarity"),
    tags$li("Severity-anchored therapy recommendations using weighted improvement rates")
  ),
  
  p("Purpose: CF predicts which therapy is most likely to help a new patient by comparing them to others with 
     similar symptoms and outcome histories."),
  
  h3("4. Integrated Vision"),
  
  p("Although CNN and CF were implemented separately in Python and R, the conceptual end-to-end system is:"),
  tags$ul(
    tags$li("Emotion Image → CNN → Severity Estimate → CF → Personalized Therapy Recommendation")
  ),
  
  p("This reflects a realistic AI mental-health workflow:
    • CNN identifies emotional indicators  
    • CF personalizes treatment planning  
    • Together, they provide a more objective, data-driven support tool"),
  
  br(),
  actionButton("go_to_app", "Next Page")
)

results <- fluidPage(
  h1("CNN Model Performance"),
  h2("CNN Training & Validation Accuracy"),
  plotOutput("cnn_accuracy_plot"),
  br(),
  p("Training accuracy improves steadily from ~14% to ~32%, while validation accuracy also rises to ~28%. Although validation accuracy increases more slowly due to the limited test set, both curves show a clear upward trend, 
     indicating the CNN is learning meaningful features. Compared to the random baseline of 14% for seven classes, 
     the model achieves nearly double the chance accuracy, proving it learned useful emotion patterns. 
     Extending training to 50 epochs with augmentation improved performance, 
     reaching a maximum of 32% training and 28% validation accuracy."),
           
  hr(),
           
  h2("CNN Training & Validation Loss"),
  plotOutput("cnn_loss_plot"),
  br(),
  p("Training loss decreases consistently from ~1.97 to ~1.58 over 50 epochs, while validation loss drops early 
     to ~1.88–1.92, then plateaus with small fluctuations. After about epoch 20, a gap forms between training and validation loss, 
     showing the model continues to learn the training samples while gaining limited additional generalization. 
     This pattern indicates moderate overfitting caused by the small dataset, 
     but the stable validation loss (no sharp increase) suggests the model still generalizes to some extent. 
    Overall, the CNN learns useful features early, then reaches its performance limit under current data constraints."),

  hr(),

  h1("Therapy Recommendation System"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("severity", "Enter your severity score (0–10):", min = 0, max = 10, value = 5, step = 0.1),
      actionButton("submit", "Get Recommendation")
    ),
    mainPanel(
      plotlyOutput("OutcomePlot", height = "500px"),
      hr(),
      h3("Recommended Therapy:"),
      verbatimTextOutput("therapy_result")
    )
  )
)

server <- function(input, output, session) {
  rv <- reactiveValues(page = "landing")
  
  observeEvent(input$go_to_app, {
    rv$page <- "main"
  })
  
  output$page <- renderUI({
    if (rv$page == "landing") landing_page else results
  })
  
  recommendation <- eventReactive(input$submit, {
    rec <- recommend_therapy(input$severity,
                             severity_vector,
                             therapy_matrix,
                             sim_hybrid_scaled,
                             k = 8)
    rec$best_therapy  
  })
  
  output$therapy_result <- renderText({
    score <- input$severity
    therapy <- recommendation()
    paste0("Severity score: ", round(score, 2),
           "\nRecommended therapy: ", therapy)
  })
  
  # Interactive plot
  output$OutcomePlot <- renderPlotly({
    interactive_plot
  })
  
  output$cnn_accuracy_plot <- renderPlot({
    img <- readPNG("Accuracy.png")
    grid.raster(img)
  })
  
  output$cnn_loss_plot <- renderPlot({
    img <- readPNG("Loss.png")
    grid.raster(img)
  })
}

ui <- fluidPage(uiOutput("page"))
shinyApp(ui = ui, server = server)