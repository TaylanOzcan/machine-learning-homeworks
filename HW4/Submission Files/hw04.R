#######################################################
## Title: Comp421 Hw04: Nonparametric Regression
## Date: November 08, 2018
## Author: Ozgur Taylan Ozcan
#######################################################

# Read the data from csv file
data_set <- read.csv("hw04_data_set.csv")

# Get x and y data
x <- data_set$x
y <- data_set$y

# Set the number of training and test data
N_train = 100
N_test = 33
N = N_train + N_test

# Get x and y training data
x_train <- x[1:N_train]
y_train <- y[1:N_train]

# Get x and y test data
x_test <- x[(N_train + 1):(N_train + N_test)]
y_test <- y[(N_train + 1):(N_train + N_test)]

point_colors <- c("blue", "red")
z <- c(rep(1, N_train), rep(2, N_test))
minimum_x <- min(x)
maximum_x <- max(x)
minimum_y <- min(y)
maximum_y <- max(y)
data_interval <- seq(from = 0, to = 60, by = 0.01)


##########################################################################


# Regressogram function
regressogram <- function(b) {
  sum = 0
  count = 0
  for (i in 1:length(x_train)) {
    if (left_borders[b] < x_train[i] & x_train[i] <= right_borders[b]) {
      sum = sum + y_train[i]
      count = count + 1
    }
  }
  average <- sum / count
  return (average)
  
}

# Plot data for regressogram
bin_width <- 3
left_borders <- seq(from = 0, to = 60 - bin_width, by = bin_width)
right_borders <- seq(from = 0 + bin_width, to = 60, by = bin_width)
p_head <- sapply(1:length(left_borders), regressogram)

plot(x, y, type = "p", pch = 19, col = point_colors[z],
     ylim = c(minimum_y, maximum_y), xlim = c(minimum_x - 3, maximum_x + 3),
     ylab = "y", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))

legend("topright", legend = c("training", "test"),
       col = point_colors, pch = 19)

for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]), c(p_head[b], p_head[b]), col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]), c(p_head[b], p_head[b + 1]), col = "black")
  }
}

# RMSE for Regressogram
rmse <- 0
for (i in 1:N_test) {
  rmse <- rmse + (y_test[i] - p_head[ceiling(x_test[i] / 3)]) ^ 2
}
rmse <- sqrt(rmse / N_test)
sprintf("Regressogram => RMSE is %g when h is %g", rmse, bin_width)



##########################################################################


# Running Mean Smoother function
rms <- function(b) {
  sum <- 0
  count <- 0
  for (i in 1:length(x_train)) {
    if (abs((b - x_train[i]) / 3) < 1) {
      sum <- sum + y_train[i]
      count <- count + 1
    }
  }
  average <- sum / count
  return (average)
}

# Plot data for Running Mean Smoother
bin_width <- 3
p_head <- sapply(1:length(data_interval), rms)

plot(data_interval, p_head, type="l", col="black",
     ylim = c(minimum_y, maximum_y), xlim = c(minimum_x - 3, maximum_x + 3),
     ylab = "y", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))

points (x, y, type = "p", pch = 19, col = point_colors[z])

legend("topright", legend = c("training", "test"),
       col = point_colors, pch = 19)


# RMSE for Running Mean Smoother
rmse <- 0
for (i in 1:N_test) {
  rmse <- rmse + (y_test[i] - p_head[x_test[i]])^ 2
}
rmse <- sqrt(rmse / N_test)
sprintf("Running Mean Smoother => RMSE is %g when h is %g", rmse, bin_width)



##########################################################################


bin_width <- 1

# Kernel Smoother
kernel <- function(b) {
  sum_k <- 0
  sum_ky <- 0
  for (i in 1:length(x_train)) {
    u <- (b - x_train[i]) / bin_width
    k <- 1 / sqrt(2*pi) * exp(-1 * u^2 / 2)
    sum_k <- sum_k + k
    sum_ky <- sum_ky + (k * x_train[i])
  }
  result <- sum_ky / sum_k
  return (result)
}

# Plot data for Kernel Smoother
left_borders <- seq(from = 0, to = 59, by = 1)
right_borders <- seq(from = 1, to = 60, by = 1)
p_head <- sapply(1:length(left_borders), kernel)

plot(x, y, type = "p", pch = 19, col = point_colors[z],
     ylim = c(minimum_y, maximum_y), xlim = c(minimum_x - 3, maximum_x + 3),
     ylab = "y", xlab = "x", las = 1, main = sprintf("h = %g", bin_width))

legend("topright", legend = c("training", "test"),
       col = point_colors, pch = 19)

for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]), c(p_head[b], p_head[b]), col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]), c(p_head[b], p_head[b + 1]), col = "black")
  }
}

# RMSE for Kernel Smoother
rmse <- 0
for (i in 1:N_test) {
  rmse <- rmse + (y_test[i] - p_head[x_test[i]])^ 2
}
rmse <- sqrt(rmse / N_test)
sprintf("Kernel Smoother => RMSE is %g when h is %g", rmse, bin_width)