# MAXIMUM LOG LIKELIHOOD ASSIGNMENT
# CREATED BY - Aditya Vyas, Vedant Choudhary
library(Rlab)
library(tidyverse)
install.packages('pracma')
library(pracma)

########################################################################################
######### MAXIMUM LIKELIHOOD ESTIMATION FUNCTIONS FOR DIFFERENT DISTRIBUTIONS ##########
########################################################################################

# Bernoulli Distribution
mle_bernoulli <- function(input_data){
  estimated_p <- mean(input_data)
  return(estimated_p)
}

# Binomial Distribution
mle_binomial <- function(input_data){
  n <- length(input_data)
  estimated_p <- (1/length(input_data))*(sum(input_data)/n)
  return(estimated_p)
}

# Geometric Distribution
mle_geometric <- function(input_data){ 
  estimated_p <- 1.0/(mean(input_data))
  return(estimated_p)
}

# Poisson Distribution
mle_poisson <- function(input_data){ 
  estimated_lambda <- mean(input_data)
  return(estimated_lambda)
}

# Uniform Distribution
mle_uniform <- function(input_data){
  estimated_a <- min(input_data)
  estimated_b <- max(input_data)
  return(c(estimated_a, estimated_b))
}

# Normal Distribution
mle_normal <- function(input_data){
  # Estimating the parameters
  estimated_mu <- mean(input_data)
  estimated_variance <- sum((input_data - estimated_mu)**2)/(length(input_data) - 1)
  return(c(estimated_mu, estimated_variance))
}

# Exponential Distribution
mle_exponential <- function(input_data){
  estimated_theta <- mean(input_data)
  return(estimated_theta)
}

# Gamma Distribution
mle_gamma <- function(input_data){
  # There are two parameters in the Gamma distribution - alpha and beta. Estimation of beta is - 
  #
  #     beta = mean(input_data)/alpha
  #
  # So, first we need to approximate the alpha parameter. Alpha is approximated using the following equation - 
  #
  #     ln(alpha) - digamma(alpha) = ln(mean(input_data)) - mean(log(input_data))
  #
  # There is no close formed solution for alpha, so we approximate it using the following approximation - 
  #
  #     ln(alpha) - digamma(alpha) ~ (1/2*alpha)(1 + 1/(6*alpha + 1))
  #
  # If we take 
  #
  #     s = ln(mean(input_data)) - mean(log(input_data))
  #
  # Then, we approximate alpha using the following solution -
  #
  #     alpha ~ (3 - s + sqrt((s-3)**2 + 24*s))/12*s
  #
  # And then using this value of alpha we can easily estimate the beta value
  
  input_data <- input_data + 1e-6
  s = log(mean(input_data)) - (sum(log(input_data)))/length(input_data)
  estimated_alpha <- ((3 - s) + sqrt( ((s-3)**2) + (24*s) ))/(12*s)
  estimated_beta <- mean(input_data)/estimated_alpha
  return(c(estimated_alpha, estimated_beta))
}

# Beta Distribution
mle_beta <- function(input_data){
  input_data_mean <- mean(input_data)
  input_data_variance <- (sum(input_data * input_data))/length(input_data)
  alpha <- ((input_data_mean ^ 2) - (input_data_mean * input_data_variance))/(input_data_variance - (input_data_mean ^ 2))
  beta <- (alpha * (1 - input_data_mean))/(input_data_mean)
  
  final_val <- c(alpha, beta)

  # We will run the optimisation step for 100 iterations
  for(index in 1:100){
    g1 <- digamma(alpha) - digamma(alpha + beta) - (sum(log(input_data)))/length(input_data)
    g2 <- digamma(beta) - digamma(alpha + beta) - (sum(log(1 - input_data))/length(input_data))
    g <- c(g1, g2)
    
    G1_val <- trigamma(alpha) - trigamma(alpha + beta)
    G2_val <- -trigamma(alpha + beta)
    G3_val <- trigamma(beta) - trigamma(alpha + beta)
    G <- matrix(c(G1_val, G2_val, G2_val, G3_val), nrow = 2, ncol = 2, byrow = TRUE)
    G_inverse <- inv(G)
    
    # Final values
    final_val <- final_val - t(G_inverse %*% g)
    alpha <- final_val[1]
    beta <- final_val[2]
  }
  
  return(c(c(alpha, beta)))
}

# Chi Square Distribution
mle_chisq <- function(input_data){
  # Intitial values for v from MOM estimator
  p_tilda <- mean(input_data)
  
  # We will use some approximations using the second derivative
  n <- length(input_data)
  del_p_numerator <- (-n/gamma(p_tilda/2) * digamma(p_tilda/2)) - (((n * log(2)) + sum(log(input_data)))/2)
  del_p_denominator <- (-n * trigamma(p_tilda/2)/4)
  del_p <- del_p_numerator/del_p_denominator
  
  estimated_p <- (p_tilda + del_p)/2
  return(estimated_p)
}

# Parametric Bootstrap using KS Test
gfit <- function(distribution, nboot = 1000, input_data){
  mle_name = get(paste("mle_", distribution, sep = ""))
  theta_hat = mle_name(input_data)
  n <- length(input_data)
  
  if(distribution == "poisson"){
    q_hat <- qpois(c(1:n)/(n+1),theta_hat)
    
    D0 <- ks.test(input_data, q_hat)$statistic
    Dvec<-NULL
    
    for(i in 1:nboot){
      x_star <- rpois(n, theta_hat)
      theta_hat_star <- mle_name(x_star)
      
      q_hat_star <- qpois(c(1:n)/(n+1), theta_hat_star)
      D_star <- ks.test(x_star, q_hat_star)$statistic
      Dvec <- c(Dvec, D_star)
    }
    p_value <- sum(Dvec > D0)/nboot
    return(p_value)
  }
  else if(distribution == "normal"){
    q_hat <- qnorm(c(1:n)/(n+1),mean = theta_hat[1], sd = theta_hat[2])
    
    D0 <- ks.test(input_data, q_hat)$statistic
    Dvec<-NULL
    
    for(i in 1:nboot){
      x_star <- rnorm(n,mean = theta_hat[1], sd =theta_hat[2])
      theta_hat_star <- mle_name(x_star)
      
      q_hat_star <- qnorm(c(1:n)/(n+1),mean = theta_hat_star[1], sd =theta_hat_star[2])
      D_star <- ks.test(x_star, q_hat_star)$statistic
      Dvec <- c(Dvec, D_star)
    }
    p_value <- sum(Dvec > D0)/nboot
    return(p_value)
  }
  else if(distribution == "uniform"){
    q_hat <- qunif(c(1:n)/(n+1), theta_hat[1], theta_hat[2])
    
    D0 <- ks.test(input_data, q_hat)$statistic
    Dvec<-NULL
    
    for(i in 1:nboot){
      x_star <- runif(n, theta_hat[1], theta_hat[2])
      theta_hat_star <- mle_name(x_star)
      
      q_hat_star <- qunif(c(1:n)/(n+1), theta_hat_star[1], theta_hat_star[2])
      D_star <- ks.test(x_star, q_hat_star)$statistic
      Dvec <- c(Dvec, D_star)
    }
    p_value <- sum(Dvec > D0)/nboot
    return(p_value)
  }
  else if(distribution == "gamma"){
    q_hat <- qgamma(c(1:n)/(n+1), shape = theta_hat[1],  scale = theta_hat[2])
    
    D0 <- ks.test(input_data, q_hat)$statistic
    Dvec<-NULL
    
    for(i in 1:nboot){
      x_star <- rgamma(n, shape = theta_hat[1], scale = theta_hat[2])
      theta_hat_star <- mle_name(x_star)
      
      q_hat_star <- qgamma(c(1:n)/(n+1), shape = theta_hat_star[1], scale = theta_hat_star[2])
      D_star <- ks.test(x_star, q_hat_star)$statistic
      Dvec <- c(Dvec, D_star)
    }
    p_value <- sum(Dvec > D0)/nboot
    return(p_value)
  }
  else if(distribution == "beta"){
    q_hat <- qbeta(c(1:n)/(n+1),shape1 = theta_hat[1], shape2 = theta_hat[2])
    
    D0 <- ks.test(input_data, q_hat)$statistic
    Dvec<-NULL
    
    for(i in 1:nboot){
      x_star <- rbeta(n, shape1 =  theta_hat[1],shape2 =  theta_hat[2])
      theta_hat_star <- mle_name(x_star)
      
      q_hat_star <- qbeta(c(1:n)/(n+1), shape1 =  theta_hat_star[1], shape2 = theta_hat_star[2])
      D_star <- ks.test(x_star, q_hat_star)$statistic
      Dvec <- c(Dvec, D_star)
    }
    p_value <- sum(Dvec > D0)/nboot
    return(p_value)
  }
  else if(distribution == "exponential"){
    q_hat <- qexp(c(1:n)/(n+1),theta_hat)
    
    D0 <- ks.test(input_data, q_hat)$statistic
    Dvec<-NULL
    
    for(i in 1:nboot){
      x_star <- rexp(n, theta_hat)
      theta_hat_star <- mle_name(x_star)
      
      q_hat_star <- qexp(c(1:n)/(n+1), theta_hat_star)
      D_star <- ks.test(x_star, q_hat_star)$statistic
      Dvec <- c(Dvec, D_star)
    }
    p_value <- sum(Dvec > D0)/nboot
    return(p_value)
  }
}

######################################################################################################
########### MAIN CALL FUNCTION FOR MAXIMUM LIKELIHOOD FUNCTIONS AND PARAMETRIC BOOTSTRAP #############
######################################################################################################
# Input - Distribution name
#         Population - If user does not want to insert a population, in-built populations will be used
#         User can also change in-built populations according to his/her need
mle_wrapper <- function(distribution, population = 0){
  p = 0.5
  lambda = 0.5
  a = 0
  b = 100
  theta = 2
  alpha = 4.7
  beta = 2.9
  dog = 5
  
  if (distribution == "bernoulli"){
    if (population == 0){
      p = 0.5
      input_data = rbinom(10000, 1, p)  
    }
    print("Population parameter: ")
    print(p)
    sample_data = sample(input_data, 1000)
    parameter_estimates <- mle_bernoulli(sample_data)
    print("Parameter Estimates: ")
    print(parameter_estimates)
  }
  else if (distribution == "binomial"){
    if (population == 0){
      n = 1000
      p = 0.5
      input_data = rbinom(10000, n, p)  
    }
    print("Population parameters: ")
    print(paste(p,",",n))
    sample_data = sample(input_data, 1000)
    parameter_estimates <- mle_binomial(sample_data)
    print("Parameter Estimates: ")
    print(parameter_estimates)
  }
  else if (distribution == "geometric"){
    if (population == 0){
      p = 0.5
      input_data = rgeom(10000, p)  
    }
    print("Population parameters: ")
    print(p)
    sample_data = sample(input_data, 1000)
    parameter_estimates <- mle_geometric(sample_data)
    print("Parameter Estimates: ")
    print(parameter_estimates)
  }
  else if (distribution == "poisson"){
    if (population == 0){
      lambda = 0.5
      input_data = rpois(10000, lambda)  
    }
    print("Population parameters: ")
    print(lambda)
    sample_data = sample(input_data, 1000)
    parameter_estimates <- mle_poisson(input_data)
    print("Parameter Estimates: ")
    print(parameter_estimates)
    
    # Doing parametric bootstrap of MLE using ks test
    p_value <- gfit(distribution, input_data = input_data)
    print("The p-value is: ")
    print(p_value)
  }
  else if (distribution == "uniform"){
    if (population == 0){
      a = 0
      b = 100
      input_data = runif(10000,a,b)  
    }
    print("Population parameters: ")
    print(paste(a,",",b))
    sample_data = sample(input_data, 1000)
    estimator <- mle_uniform(sample_data)
    print("Parameter Estimates: ")
    print(estimator)
    
    # Doing parametric bootstrap of MLE using ks test
    p_value <- gfit(distribution, input_data = input_data)
    print("The p-value is: ")
    print(p_value)
  }
  else if (distribution == "normal"){
    if (population == 0){
      input_data = rnorm(10000, 0, 1)  
    }
    print("Population mean: ")
    print(mean(input_data))
    print("Population variance: ")
    print(var(input_data))
    sample_data = sample(input_data, 1000)
    parameter_estimates <- mle_normal(sample_data)
    print("Parameter Estimates: ")
    print(parameter_estimates)
    
    # Doing parametric bootstrap of MLE using ks test
    p_value <- gfit(distribution, input_data = input_data)
    print("The p-value is: ")
    print(p_value)
  }
  else if (distribution == "exponential"){
    if (population == 0){
      theta = 2
      input_data = rexp(10000, theta)
    }
    print("Population parameter: ")
    print(theta)
    sample_data = sample(input_data, 1000)
    parameter_estimates <- mle_exponential(input_data = sample_data)
    print("Parameter Estimates: ")
    print(parameter_estimates)
    
    # Doing parametric bootstrap of MLE using ks test
    p_value <- gfit(distribution, input_data = input_data)
    print("The p-value is: ")
    print(p_value)
  }
  else if (distribution == "gamma"){
    if (population == 0){
      alpha = 5
      beta = 20
      input_data = rgamma(10000, shape = alpha, scale = beta)  
    }
    print("Population parameters: ")
    print(paste(alpha,",",beta))
    sample_data = sample(input_data, 1000)
    parameter_estimates <- mle_gamma(sample_data)
    print("Parameter Estimates: ")
    print(parameter_estimates)
    
    # Doing parametric bootstrap of MLE using ks test
    p_value <- gfit(distribution, input_data = input_data)
    print("The p-value is: ")
    print(p_value)
  }
  else if (distribution == "beta"){
    if (population == 0){
      alpha = 4.7
      beta = 2.9
      input_data = rbeta(10000, shape1 = alpha, shape2 = beta)  
    }
    print("Population parameters: ")
    print(paste(alpha,",",beta))
    sample_data = sample(input_data, 1000)
    parameter_estimates <- mle_beta(sample_data)
    print("Parameter Estimates: ")
    print(parameter_estimates)
    
    # Doing parametric bootstrap of MLE using ks test
    p_value <- gfit(distribution, input_data = input_data)
    print("The p-value is: ")
    print(p_value)
  }
  else if (distribution == "chi square"){
    if (population == 0){
      dog = 5
      input_data = rchisq(10000, df = dog)  
    }
    print("Population parameter: ")
    print(dog)
    sample_data = sample(input_data, 1000)
    parameter_estimates <- mle_chisq(sample_data)
    print("Parameter Estimates: ")
    print(parameter_estimates)
  }
}

# Valid distributions for the program
repeat{
  cat("Valid Distributions: \n1. Bernoulli\n2. Binomial\n3. Geometric\n4. Poisson (gof available)\n5. Uniform (gof available)\n6. Normal (gof available)\n7. Exponential (gof available)\n8. Gamma (gof available)\n9. Beta (gof available)\n10. Chi-Square")
  distribution <- readline(prompt = "For which distribution, do you want M.L.E. estimation (write 'exit' to exit): ")
  mle_wrapper(distribution)
  
  if(distribution == "exit")
    break;
}


