Jackknife <- function(v, statfunc = sd){
  
  # Calculate length of a vector v1
  n <- length(v)
  
  # Initialize an empty vector
  jackvec <- c()
  
  # Calculate standard deviation of v
  sd0 <- statfunc(v)
  
  # Calculate the mean of v
  mu0 <- mean(v)
  
  for(i in 1:n){
    
    # Find out standard deviation/or any stat function of all values in the vector except the ith element.
    # eg. for a vector c(1,2,3,4), sd(v[-1]) will calculate the standard deviation
    # of c(2,3,4).
    sda <- statfunc(v[-i])
    
    # Calculate n*sigma for the new vector, subtract this from the original vector and append it in the list.
    jackvec <- c(jackvec, n*(sd0) - (n - 1)*sda)
  }
  
  jackbias <- mean(jackvec) - sd0
  jacksd <- sd(jackvec)/sqrt(n)
  
  list(mu0 = mu0, jackbias = jackbias, jacksd = jacksd)
}

BootstrapBias <- function(v, statfunc = sd){
  # Calculate length of a vector v1
  n <- length(v)
  
  # Initialize an empty vector
  bootvec <- c()
  
  # Calculate standard deviation of v
  sd0 <- statfunc(v)
  
  # Calculate the mean of v
  mu0 <- mean(v)
  
  for(i in 1:n){
    
    # Find out standard deviation/or any stat function of all values in the vector except the ith element.
    # eg. for a vector c(1,2,3,4), sd(v[-1]) will calculate the standard deviation
    # of c(2,3,4).
    sda <- statfunc(sample(v, replace = T))
    
    # Calculate n*sigma for the new vector, subtract this from the original vector and append it in the list.
    bootvec <- c(bootvec, n*(sd0) - (n - 1)*sda)
  }
  
  bootbias <- mean(bootvec) - sd0
  bootsd <- sd(bootvec)/sqrt(n)
  
  list(mu0 = mu0, bootbias = bootbias, bootsd = bootsd)
}

Jackknife(c(1,2,3,4,5,6,7,8,9,10))
