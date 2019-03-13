##################################################
## Title: Comp421 Hw01: Naive Bayes Classifier
## Date: October 15, 2018
## Author: Ozgur Taylan Ozcan
##################################################

labelsFile <- "hw01_data_set_labels.csv"
imagesFile <- "hw01_data_set_images.csv"

# Create template confusion matrices (5x5)
training_confusion_matrix <- matrix(0, 5, 5)
test_confusion_matrix <- matrix(0, 5, 5)

# Read training data from file (first 25 images from each class)
trainingA <- read.csv(imagesFile, header=FALSE, skip=0, nrows=25)
trainingB <- read.csv(imagesFile, header=FALSE, skip=39, nrows=25)
trainingC <- read.csv(imagesFile, header=FALSE, skip=39*2, nrows=25)
trainingD <- read.csv(imagesFile, header=FALSE, skip=39*3, nrows=25)
trainingE <- read.csv(imagesFile, header=FALSE, skip=39*4, nrows=25)
# Create the training set
trainingSet <- as.matrix(rbind(trainingA, trainingB, trainingC, trainingD, trainingE))

# Read test data from file (remaining 14 images from each class)
testA <- read.csv(imagesFile, header=FALSE, skip=0 + 25, nrows=14)
testB <- read.csv(imagesFile, header=FALSE, skip=39 + 25, nrows=14)
testC <- read.csv(imagesFile, header=FALSE, skip=39*2 + 25, nrows=14)
testD <- read.csv(imagesFile, header=FALSE, skip=39*3 + 25, nrows=14)
testE <- read.csv(imagesFile, header=FALSE, skip=39*4 + 25, nrows=14)
# Create the test set
testSet <- as.matrix(rbind(testA, testB, testC, testD, testE))

# Estimate the parameters by taking means of training data for each class, and create the pcd
pcd <- t(as.matrix(rbind(colMeans(trainingA), colMeans(trainingB), colMeans(trainingC), colMeans(trainingD), colMeans(trainingE))))

# The safe log function
modified_log <- function(val){
  return (log(val + 1e-100))
}

# The prediction function which predicts the class a given image might belong to
predict <- function(image){
  log_pcd <- apply(pcd, MARGIN=c(1, 2), FUN=modified_log)
  log_one_minus_pcd <- apply(1 - pcd, MARGIN=c(1, 2), FUN=modified_log)
  
  scoreA <- ( sum(log_pcd[,1] * image) + sum(log_one_minus_pcd[,1] * (1 - image)) )
  scoreB <- ( sum(log_pcd[,2] * image) + sum(log_one_minus_pcd[,2] * (1 - image)) )
  scoreC <- ( sum(log_pcd[,3] * image) + sum(log_one_minus_pcd[,3] * (1 - image)) )
  scoreD <- ( sum(log_pcd[,4] * image) + sum(log_one_minus_pcd[,4] * (1 - image)) )
  scoreE <- ( sum(log_pcd[,5] * image) + sum(log_one_minus_pcd[,5] * (1 - image)) )
  
  scores <- c(scoreA, scoreB, scoreC, scoreD, scoreE)
  
  return(which.max(scores))
}

# Loop for filling the confusion matrix for training data
for (i in 1:nrow(trainingSet)) {
  predicted <- predict(trainingSet[i,])
  actual <- floor((i-1)/25) + 1
  training_confusion_matrix[predicted, actual] <- training_confusion_matrix[predicted, actual] + 1
}

# Loop for filling the confusion matrix for test data
for (i in 1:nrow(testSet)) {
  predicted <- predict(testSet[i,])
  actual <- floor((i-1)/14) + 1
  test_confusion_matrix[predicted, actual] <- test_confusion_matrix[predicted, actual] + 1
}

# Print confusion matrices for training and test data
training_confusion_matrix
test_confusion_matrix

