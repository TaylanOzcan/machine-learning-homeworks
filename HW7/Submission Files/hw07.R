#####################################################################
## Title: Comp421 Hw07: Modeling Outbound Reach Rate of a Call Center
## Date: January 09, 2019
## Author: Ozgur Taylan Ozcan
#####################################################################

library(e1071)
library(MASS)
library(caret)
library(cvAUC)
library(AUC)
library(tree)


#### Read data  #####################################################

X_train <- read.csv("training_data.csv", header = TRUE)
X_test <- read.csv("test_data.csv", header = TRUE)
y_train <- as.factor(read.csv("training_labels.csv", header = FALSE)[,1])


#### Preprocess - PCA  ##############################################

set.seed(521)
pre_process <- preProcess(X_train, method = c("center", "scale", "pca", "zv"), thresh = 0.95)
X_train_processed <- predict(pre_process, X_train)
X_test_processed <- predict(pre_process, X_test)

data_processed <-  cbind(X_train_processed, y = y_train)


#### Multiple Learner Functions #####################################

multiple_learners_train <- function(data_train){
  # Train each learner
  tree_classifier <-  tree(y ~ ., data=data_train)
  log_reg_classifier <- glm(y ~ ., data=data_train, family=binomial())
  lda_classifier <- lda(y ~ ., data=data_train)
  
  # Return list of learners along with their weight
  return(list(list(tree_classifier, 0.25),
              list(log_reg_classifier, 0.4),
              list(lda_classifier, 0,35))
  )
}

multiple_learners_predict <- function(base_learners, data_test){
  # Get predictions for each learner
  training_scores_tree <- predict(base_learners[[1]][[1]], data_test)[,2]
  training_scores_log_reg <- predict(base_learners[[2]][[1]], data_test, type = "response")
  training_scores_lda <- predict(base_learners[[3]][[1]], data_test)$posterior[,2]
  
  # Combine scores by summing weighted scores for each learner
  combined_scores <- 
    (training_scores_tree * base_learners[[1]][[2]]) + 
    (training_scores_log_reg * base_learners[[2]][[2]]) + 
    (training_scores_lda * base_learners[[3]][[2]])
  
  return(combined_scores)
}


#### Cross Validation Functions #####################################

.cv_get_folds <- function(Y, K){
  Y0 <- split(sample(which(Y==0)), rep(1:K, length=length(which(Y==0))))
  Y1 <- split(sample(which(Y==1)), rep(1:K, length=length(which(Y==1))))
  folds <- vector("list", length=K)
  for (k in seq(K)) {folds[[k]] <- c(Y0[[k]], Y1[[k]])}     
  return(folds)
}

.cv_fit <- function(k, folds, data){
  multiple_learners_fit <- multiple_learners_train(data[-folds[[k]], ])
  prediction <- multiple_learners_predict(multiple_learners_fit, data[folds[[k]], ])
  return(prediction)
}

k_fold_cv <- function(data, K = 10){
  folds <- .cv_get_folds(Y = data[, 'y'], K = K)
  predictions <- c()
  for(k in 1:K){
    predictions <- c(predictions, .cv_fit(k, folds, data))
    cat("K-Fold Cross Validation: Iteration ", k, "/", K, "\n")
  }
  predictions[unlist(folds)] <- predictions
  cv_result <- ci.cvAUC(predictions = predictions, 
                        labels = data[, 'y'],
                        folds = folds, 
                        confidence = 0.95)
  roc_curve <- roc(predictions = predictions, labels = data[,'y'])
  auc(roc_curve)
  plot(roc_curve$fpr, roc_curve$tpr, col = "green", lwd = 2, type = "b", las = 1,
       xlab="FP", ylab="TP", main="ROC for Cross Validation")
  return(cv_result)
}

five_x_two_cv <- function(data){
  folds <- .cv_get_folds(Y = data[, 'y'], 10)
  predictions <- c()
  for(i in 1:5){
    for(j in 1:2){
      index1 <- (i-1)*2 + j
      index2 <- (i-1)*2 + (3-j)
      multiple_learners_fit <- multiple_learners_train(data[folds[[index1]], ])
      cur_prediction <- multiple_learners_predict(multiple_learners_fit, data[folds[[index2]], ])
      predictions <- c(predictions, cur_prediction)
    }
    cat("5x2 Cross Validation: Iteration ", i, "/", 5, "\n")
  }
  predictions[unlist(folds)] <- predictions
  cv_result <- ci.cvAUC(predictions = predictions, 
                        labels = data[, 'y'],
                        folds = folds, 
                        confidence = 0.95)
  roc_curve <- roc(predictions = predictions, labels = data[,'y'])
  auc(roc_curve)
  plot(roc_curve$fpr, roc_curve$tpr, col = "green", lwd = 2, type = "b", las = 1,
       xlab="FP", ylab="TP", main="ROC for Cross Validation")
  return(cv_result)
}


#### Apply Cross Validation on Training Data ########################

#cv <- k_fold_cv(data_processed, K = 10)
cv <- five_x_two_cv(data_processed)
print("Cross Validation completed")
print(cv)


#### Predict Scores for Test Data and Write Into File ###############

print("Training with the train data")
multiple_learners_fit <- multiple_learners_train(data_processed)
print("Predicting test data")
test_scores <- multiple_learners_predict(multiple_learners_fit, X_test_processed)
write.table(test_scores, file = "test_predictions.csv", row.names = FALSE, col.names = FALSE)
print("Test predictions were written in file")