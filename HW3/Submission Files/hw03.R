############################################################################
## Title: Comp421 Hw03: Multilayer Perceptron for Multiclass Discrimination
## Date: October 31, 2018
## Author: Ozgur Taylan Ozcan
############################################################################

labelsFile <- "hw03_data_set_labels.csv"
imagesFile <- "hw03_data_set_images.csv"

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
sigmoid_func <- function(a) {
  return (1 / (1 + exp(-a)))
}

# The softnax function
softmax_func <- function(Z, v) {
  return (exp(Z %*% t(v)) / (rowSums(exp(Z %*% t(v)))))
}

# The safelog function
safelog_func <- function(x) {
  return (log(x + 1e-100))
}

# Set the learning parameters
H <- 20
eta <- 0.005
epsilon <- 1e-3
max_iteration <- 200

set.seed(521)


################## TRAINING DATA ##################

X <- trainingSet

# Set the R matrix
R = matrix(0, nrow = 125, ncol = 5)
for(i in 1:5){
  index = ((i-1)*25 + 1)
  R[index:(index+24), i] = 1
}

# Get the number of samples and features
N <- nrow(R)
D <- ncol(X)

# Initalize W and v with uniform random values
W <- matrix(runif(D * H, min = -0.01, max = 0.01), D, H)
v <- matrix(runif((5) * H, min = -0.01, max = 0.01), 5, H)

# Initialize Z, training_Y and the objective values
Z <- sigmoid_func(X %*% W)
training_Y <- softmax_func(Z, v)
objective_values <- -sum(R * (safelog_func(training_Y)))

# Learn v & W using backpropagation and capture the objective function values
iteration <- 1
while (1) {
  for (i in sample(N)) {
    Z[i,] <- sigmoid_func(X[i,] %*% W)
    training_Y[i,] <- softmax_func(Z[i,], v)
    
    v <- v + eta * (R[i,] - training_Y[i,]) %*% t(Z[i,])
    
    for (h in 1:H) {
      W[,h] <- W[,h] + eta * sum((R[i,] - training_Y[i,]) * v[,h]) * Z[i, h] * (1 - Z[i, h]) %*% X[i,]
    }
  }
  
  Z <- sigmoid_func(X %*% W)
  training_Y <- softmax_func(Z,v)
  objective_values <- c(objective_values, -sum(R * (safelog_func(training_Y))))
  
  if (iteration >= max_iteration | epsilon > abs(objective_values[iteration + 1] - objective_values[iteration])) {
    break
  }
  
  iteration <- iteration + 1
}

# Draw the objective function
plot(
  1:(iteration+1),
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
test_Y <- test_X %*% W %*% t(v)
test_Y <- apply(test_Y, 1, which.max)
test_confusion_matrix <- table(test_Y, test_actual)

# Print the confusion matrices
print(training_confusion_matrix)
print(test_confusion_matrix)