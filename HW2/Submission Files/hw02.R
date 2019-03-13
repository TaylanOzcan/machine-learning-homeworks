##################################################
## Title: Comp421 Hw02: Discrimination by Regression
## Date: October 24, 2018
## Author: Ozgur Taylan Ozcan
##################################################

labelsFile <- "hw02_data_set_labels.csv"
imagesFile <- "hw02_data_set_images.csv"

# Create template confusion matrices (5x5)
training_confusion_matrix <- matrix(0, 5, 5)
test_confusion_matrix <- matrix(0, 5, 5)

# Read training data from file (first 25 images from each class)
trainingA <- read.csv(imagesFile, header = FALSE, skip = 0, nrows = 25)
trainingB <- read.csv(imagesFile, header = FALSE, skip = 39, nrows = 25)
trainingC <- read.csv(imagesFile, header = FALSE, skip = 39 * 2, nrows = 25)
trainingD <- read.csv(imagesFile, header = FALSE, skip = 39 * 3, nrows = 25)
trainingE <- read.csv(imagesFile, header = FALSE, skip = 39 * 4, nrows = 25)

# Create the training set
trainingSet <- as.matrix(rbind(trainingA, trainingB, trainingC, trainingD, trainingE))

# Read test data from file (remaining 14 images from each class)
testA <- read.csv(imagesFile, header = FALSE, skip = 0 + 25, nrows = 14)
testB <- read.csv(imagesFile, header = FALSE, skip = 39 + 25, nrows = 14)
testC <- read.csv(imagesFile, header = FALSE, skip = 39 * 2 + 25, nrows = 14)
testD <- read.csv(imagesFile, header = FALSE, skip = 39 * 3 + 25, nrows = 14)
testE <- read.csv(imagesFile, header = FALSE, skip = 39 * 4 + 25, nrows = 14)

# Create the test set
testSet <- as.matrix(rbind(testA, testB, testC, testD, testE))


############## FUNCTIONS / PARAMETERS #############

# The sigmoid function
sigmoid_func <- function(X, W, w0) {
  return (1 / (1 + exp(-(X %*% W + w0))))
}

# The update function for W
update_W <- function(X, R, Y) {
  return(-t(X) %*% (((R - Y) * Y * (1 - Y))))
}

# The update function for w0
update_w0 <- function(R, Y) {
  return (-colSums((R - Y) * Y * (1 - Y)))
}

# Learning parameters
epsilon <- 1e-3
eta <- 0.01

set.seed(521)


################## TRAINING DATA ##################

X <- trainingSet

# Set the R matrix
R = matrix(0, nrow = 125, ncol = 5)
for(i in 1:5){
  index = ((i-1)*25 + 1)
  R[index:(index+24), i] = 1
}

H <- 5
D <- ncol(X)

# Initalize W and w0 with uniform random values
W <- matrix(runif(D * H, min = -0.01, max = 0.01), D, H)
w0 <- runif(H, min = -0.01, max = 0.01)

objective_values <- c()

# Learn W & w0 and capture the objective function values
iteration <- 1
while (1) {
  training_Y <- sigmoid_func(X, W, w0)
  
  objective_values <- c(objective_values, sum((R - training_Y) ^ 2) * (0.5))
  
  prev_W <- W
  W <- W - eta * update_W(X, R, training_Y)
  
  prev_w0 <- w0
  w0 <- w0 - eta * update_w0(R, training_Y)
  
  if (epsilon > sqrt(sum((w0 - prev_w0) ^ 2) + sum((W - prev_W) ^ 2))) {
    break
  }
  
  iteration <- iteration + 1
}

# Draw the objective function
plot(
  1:iteration,
  objective_values,
  type = "l",
  las = 1,
  lwd = 2,
  xlab = "Iteration",
  ylab = "Error"
)

# Set the matrix representing the actual Y values for training data
training_actual = matrix(nrow = 125, ncol = 1)
for(i in 1:5){
  index = ((i-1)*25 + 1)
  training_actual[index:(index+24), 1] = i
}

# Set the confusion matrix for training data
training_Y <- apply(training_Y, 1, which.max)
training_confusion_matrix <- table(training_Y, training_actual)


################### TEST DATA #####################

test_X = testSet

# Set the matrix representing the actual Y values for test data
test_actual = matrix(nrow = 70, ncol = 1)
for(i in 1:5){
  index = ((i-1)*14 + 1)
  test_actual[index:(index+13), 1] = i
}

# Set the confusion matrix for test data
test_Y <- apply(sigmoid_func(test_X, W, w0), 1, which.max)
test_confusion_matrix <- table(test_Y, test_actual)

# Print the confusion matrices
print(training_confusion_matrix)
print(test_confusion_matrix)