###########################################################
## Title: Comp421 Hw06: Expectation-Maximization Clustering
## Date: December 14, 2018
## Author: Ozgur Taylan Ozcan
###########################################################

library(MASS)
library(mixtools)
set.seed(521)

# set means of gaussian densities
gaussian_means <- matrix(c(2.5, 2.5,
                           -2.5, 2.5,
                           -2.5, -2.5,
                           2.5, -2.5,
                           0.0,  0.0), 2, 5)

# set covariences of gaussian densities
gaussian_covs <- array(c(0.8, -0.6, -0.6, 0.8,
                         0.8, 0.6, 0.6, 0.8,
                         0.8, -0.6, -0.6, 0.8,
                         0.8, 0.6, 0.6, 0.8,
                         1.6,  0.0,  0.0, 1.6), c(2, 2, 5))

# set sizes of gaussian densities
gaussian_sizes <- c(50,50,50,50,100)

# get the total number of data points
N <- sum(gaussian_sizes)

# generate random data points and form the data matrix (X)
X <- rbind(
  mvrnorm(n = gaussian_sizes[1], mu = gaussian_means[, 1], Sigma = gaussian_covs[, , 1]),
  mvrnorm(n = gaussian_sizes[2], mu = gaussian_means[, 2], Sigma = gaussian_covs[, , 2]),
  mvrnorm(n = gaussian_sizes[3], mu = gaussian_means[, 3], Sigma = gaussian_covs[, , 3]),
  mvrnorm(n = gaussian_sizes[4], mu = gaussian_means[, 4], Sigma = gaussian_covs[, , 4]),
  mvrnorm(n = gaussian_sizes[5], mu = gaussian_means[, 5], Sigma = gaussian_covs[, , 5])
)

# plot the data points
plot(X[,1], X[,2], las = 1, pch = 19, xlim = c(-6, 6), ylim = c(-6, 6), xlab = "x1", ylab = "x2")

# set k=5 for k-means clustering
K <- 5
# initialize centroids randomly
centroids <- X[sample(1:N, K),]

# run k-means clustering for 2 iterations
cl <- kmeans(X, centroids, iter.max = 2, nstart = K)
centroids <- cl$centers
cluster <- cl$cluster

# calculate the prior probabilities
prior_prob <- numeric(K)
for(i in 1:K){
  prior_prob[i] = sum(i == cluster) / N
}

# calculate the initial covariences
init_covariences <- t(sapply(1:K, function (i) {
  cov(X[i == cluster,])
}))

# Use the centroids as the initial mean vectors in the EM algorithm
mean_vectors <- centroids

# Run the EM algorithm for 100 times
for (iteration in 1:100) {
  g_ic <- matrix(nrow=N, ncol=K)
  for (i in 1:N) {
    for (c in 1:K) {
      g_ic[i,c] <- prior_prob[c] * ((det(matrix(init_covariences[c,],2,2)))^(-0.5)) *
        exp((-0.5)* matrix((X[i,]- mean_vectors[c,]), ncol=2) %*% t((X[i,]- mean_vectors[c,]) %*% (solve(matrix(init_covariences[c,],2,2)))))
    }
  }
  
  sum_g_ic <- rep(0, N)
  for (i in 1:N) {
    sum_g_ic[i] <-sum(sapply(1:K, function (c) g_ic[i,c]))
  }
  
  h_ic <- t(apply(g_ic, 1, function(X) X/sum(X)))
  
  # update prior probabilities
  for (c in 1:K) {
    sum <- sum(h_ic[1:N, c])
    prior_prob[c] <- sum / N
  }
  
  # update mean vectors
  for (c in 1:K) {
    sum_h <- sum(h_ic[1:N, c])
    sum_f <- c(0,0)
    for (i in 1:N) {
      sum_f <- sum_f + h_ic[i,c] * X[i,]
    }
    mean_vectors[c, ] <- sum_f / sum_h
  }
  
  # update covariences
  for (c in 1:K) {
    sum_h <- sum(h_ic[1:N, c])
    sum_s <- matrix(c(0,0,0,0),2,2)
    for (i in 1:N) {
      sum_s <- sum_s + (X[i, ] - mean_vectors[c,]) %*% t((X[i, ] - mean_vectors[c,])) * h_ic[i,c]
    }
    init_covariences[c, ] <- sum_s / sum_h
  }
}

# print the mean vectors
print(mean_vectors)

# plot the clusters
colors <- c("orange", "red", "green", "blue", "purple")
plot(X[, 1], X[, 2], type = "p", pch = 19, col = colors[cluster], las = 1,
     xlim = c(-6, 6), ylim = c(-6, 6),
     xlab = "x1", ylab = "x2")

# draw the gaussian density lines
for(c in 1:K){
  ellipse(gaussian_means[,c], gaussian_covs[,,c], alpha = .05, npoints = gaussian_sizes[c], newplot = FALSE, draw = TRUE, lty=2)
  ellipse(mean_vectors[c,], matrix(init_covariences[c,], 2,2), alpha = .05, npoints = gaussian_sizes[c], newplot = FALSE, draw = TRUE)
}