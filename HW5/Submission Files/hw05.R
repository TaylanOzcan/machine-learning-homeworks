#######################################################
## Title: Comp421 Hw05: Decision Tree Regression
## Date: November 29, 2018
## Author: Ozgur Taylan Ozcan
#######################################################

# Read the data from csv file
data_set <- read.csv("hw05_data_set.csv")

# Get x and y data
x <- data_set$x
y <- data_set$y

# Set the number of training and test data
N_train <- 100
N_test <- 33
N <- N_train + N_test

# Get x and y training data
x_train <- x[1:N_train]
y_train <- y[1:N_train]

# Get x and y test data
x_test <- x[(N_train + 1):(N_train + N_test)]
y_test <- y[(N_train + 1):(N_train + N_test)]

# Find the min and max values for X and Y
minimum_x <- floor(min(x) - 3)
maximum_x <- ceiling(max(x) + 3)
minimum_y <- min(y)
maximum_y <- max(y)

# Decision tree regression algorithm with the given pre-pruning rule
decision_tree_regression <- function(P) {
  # Create needed structures
  node_splits <- c()
  node_means <- c()
  
  # Place training instances into the root node
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  # Learning algorithm
  while (1) {
    # Find nodes that need splitting
    split_nodes <- which(need_split)
    
    # Check whether we reach all terminal nodes
    if (length(split_nodes) == 0) {
      break
    }
    
    # Find best split positions for all nodes
    for (split_node in split_nodes) {
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      node_mean <- mean(y_train[data_indices])
      
      if (length(x_train[data_indices]) <= P) {
        is_terminal[split_node] <- TRUE
        node_means[split_node] <- node_mean
      } else {
        is_terminal[split_node] <- FALSE
        unique_values <- sort(unique(x_train[data_indices]))
        split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
        split_scores <- rep(0, length(split_positions))
        
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(x_train[data_indices] <= split_positions[s])]
          right_indices <- data_indices[which(x_train[data_indices] > split_positions[s])]
          total_error <- 0
          
          if (length(left_indices) > 0) {
            mean <- mean(y_train[left_indices])
            total_error <- total_error + sum((y_train[left_indices] - mean) ^ 2)
          }
          
          if (length(right_indices) > 0) {
            mean <- mean(y_train[right_indices])
            total_error <- total_error + sum((y_train[right_indices] - mean) ^ 2)
          }
          
          split_scores[s] <- total_error / (length(left_indices) + length(right_indices))
        }
        
        if (length(unique_values) == 1) {
          is_terminal[split_node] <- TRUE
          node_means[split_node] <- node_mean
          next 
        }
        
        best_split <- split_positions[which.min(split_scores)]
        node_splits[split_node] <- best_split
        
        # Create left node using the selected split
        left_indices <- data_indices[which(x_train[data_indices] < best_split)]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
        
        # Create right node using the selected split
        right_indices <- data_indices[which(x_train[data_indices] >= best_split)]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
  return(list("splits"= node_splits, "means"= node_means, "is_terminal"= is_terminal))
}

# Set the pre-pruning parameter P = 10
P <- 10

decision_tree <- decision_tree_regression(P)
node_splits <- decision_tree$splits
node_means <- decision_tree$means
is_terminal <- decision_tree$is_terminal

# Define regression function
predict <- function(dp, is_terminal, node_splits, node_means){
  index <- 1
  while (1) {
    if (is_terminal[index] == TRUE) {
      return(node_means[index])
    } else {
      if (dp <= node_splits[index]) {
        index <- index * 2
      } else {
        index <- index * 2 + 1
      }
    }
  }
}

# Plot training and test data on a figure
point_colors <- c("blue", "red")
z <- c(rep(1, N_train), rep(2, N_test))

plot(x, y, type = "p", pch = 19, col = point_colors[z],
     ylim = c(minimum_y, maximum_y), xlim = c(minimum_x, maximum_x),
     ylab = "y", xlab = "x", las = 1, main = "P = 10")

legend("topright", legend = c("training", "test"),
       col = point_colors, pch = 19, cex = 0.75)

# Draw the fit on the figure
left_borders <- seq(from = 0, to = 59.9, by = 0.1)
right_borders <- seq(from = 0.1, to = 60, by = 0.1)

for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]),
        c(predict(left_borders[b], is_terminal, node_splits, node_means), predict(left_borders[b], is_terminal, node_splits, node_means)),
        col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]),
          c(predict(left_borders[b], is_terminal, node_splits, node_means), predict(right_borders[b], is_terminal, node_splits, node_means)),
          col = "black")
  }
}

# Calculate and print RMSE for the test data
y_predicted <- rep(0, N_test)
for (i in 1:N_test) {
  y_predicted[i] <- predict(x_test[i], is_terminal, node_splits, node_means)
}
RMSE <- sqrt(sum((y_test - y_predicted) ^ 2) / length(y_test))
sprintf("RMSE is %.4f when P is %s", RMSE, P)

# Learn decision trees for P = 1 to 20
RMSE_per_P <- rep(0, 20)
for (p in 1:20) {
  decision_tree <- decision_tree_regression(p)
  node_splits <- decision_tree$splits
  node_means <- decision_tree$means
  is_terminal <- decision_tree$is_terminal
  y_predicted <- rep(0, N_test)
  for (i in 1:N_test) {
    y_predicted[i] <- predict(x_test[i], is_terminal, node_splits, node_means)
  }
  RMSE_per_P[p] <- sqrt(sum((y_test - y_predicted) ^ 2) / length(y_test))
}

# Plot P vs RMSE graph
plot(1:20, RMSE_per_P, type = "o", las = 1, pch = 1, lty = 2, xlab = "P", ylab = "RMSE")