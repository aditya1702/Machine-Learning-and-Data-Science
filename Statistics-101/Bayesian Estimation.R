# BAYESIAN ESTIMATION ASSIGNMENT
# CREATED BY - Aditya Vyas, Vedant Choudhary
library(Rlab)
library(actuar)
par(mfrow = c(2,1))

############################################################################
########### MAIN CALL FUNCTION FOR BAYESIAN ESTIMATION FUNCTIONS #############
############################################################################
# Input - Distribution name
#         Population - If user does not want to insert a population, in-built populations will be used
#         User can also change in-built populations according to his/her need
bayes_estimate_wrapper <- function(distribution, n = 10000){
  if (distribution == "binomial"){
    sample <- rbinom(n = 1000, size = 1, prob = 0.6)
    
    # Assuming alpha and beta for the prior distribution to be 1
    prior_alpha <- 1
    prior_beta <- 1
    r <- 1
    
    # Getting the posterior distribution parameters
    posterior_alpha <- prior_alpha + sum(sample)
    posterior_beta <- prior_beta + r*length(sample) - sum(sample)
    print("The parameters of the posterior beta distribution are:")
    print(c(posterior_alpha, posterior_beta))
    
    posterior_distribution_sample <- rbeta(n, posterior_alpha, posterior_beta)
    plot(density(posterior_distribution_sample))
  }
  else if (distribution == "poisson"){
    sample <- rpois(n = 1000, lambda = 5)
    
    # Assuming alpha and beta for the prior distribution to be 1
    prior_alpha <- 1
    prior_beta <- 1
    
    # Getting the posterior distribution parameters
    posterior_alpha <- prior_alpha + sum(sample)
    posterior_beta <- 1/(1/prior_beta + length(sample))
    print("The parameters of the posterior gamma distribution are:")
    print(c(posterior_alpha, posterior_beta))
    
    posterior_distribution_sample <- rgamma(n, posterior_alpha, posterior_beta)
    plot(density(posterior_distribution_sample))
  }
  else if (distribution == "uniform"){
    sample <- runif(n = 1000, min = 0, max = 10)
    
    # Assuming alpha and beta for the prior distribution to be 1
    prior_w0 <- 1
    prior_alpha <- 1
    
    # Getting the posterior distribution parameters
    posterior_w0 <- max(c(prior_w0, sample))
    posterior_alpha <- prior_alpha + length(sample)
    print("The parameters of the posterior pareto distribution are:")
    print(c(posterior_w0, posterior_alpha))
    
    posterior_distribution_sample <- rpareto(n, posterior_w0, posterior_alpha)
    plot(density(posterior_distribution_sample))
  }
  else if (distribution == "normal"){
    sample <- rnorm(n = 1000, mean = 10, sd = 20)
    
    # Assuming alpha and beta for the prior distribution to be 1
    r <- 1
    tau <- 5
    mu <- 4
    prior_alpha <- 1
    prior_beta <- 2
    
    # Getting the posterior distribution parameters
    M_conditional_distribution_mu <- (tau*mu + length(sample)*mean(sample))/(tau + length(sample))
    M_conditional_distribution_precision <- (tau + length(sample))*r
    print("The parameters of the conditional posterior normal distribution of M when R=r is:")
    print(c(M_conditional_distribution_mu, M_conditional_distribution_precision))
    
    R_marginal_distribution_alpha <- prior_alpha + length(sample)/2
    R_marginal_distribution_beta <- prior_beta + 1/2*(sum((sample - mean(sample))**2)) + tau*length(sample)*((mean(sample) - mu)**2)/2*(tau + length(sample))
    print("The parameters of the marginal posterior gamma distribution of R is:")
    print(c(R_marginal_distribution_alpha, R_marginal_distribution_beta))
    
    # Generate the distibutions
    conditional_joint_distribution_of_M <- rnorm(n, mean = M_conditional_distribution_mu, 1/sqrt(M_conditional_distribution_precision))
    marginal_joint_distribution_of_R <- rgamma(n, R_marginal_distribution_alpha, R_marginal_distribution_beta)
    
    #par(mfrow = (2,1))
    plot(density(conditional_joint_distribution_of_M))
    plot(density(marginal_joint_distribution_of_R))
  }
  else if (distribution == "exponential"){
    sample <- rexp(n = 1000, theta = 10)
    
    # Assuming alpha and beta for the prior distribution to be 1
    prior_alpha <- 1
    prior_beta <- 1
    
    # Getting the posterior distribution parameters
    posterior_alpha <- prior_alpha + length(sample)
    posterior_beta <- 1/(1/prior_beta + sum(sample))
    print("The parameters of the posterior gamma distribution are:")
    print(c(posterior_alpha, posterior_beta))
    
    posterior_distribution_sample <- rgamma(n, posterior_alpha, posterior_beta)
    plot(density(posterior_distribution_sample))
  }
}

# STATEMENTS TO BE EXECUTED BEFORE RUNNING THE MAIN BAYESIAN ESTIMATION FUNCTION

# Valid distributions for the program
repeat{
  cat("Valid Distributions: \n1. Binomial\n2. Poisson\n3. Uniform\n4. Normal\n5. Exponential")
  distribution <- readline(prompt="For which distribution, do you want Bayesian estimation (write 'quit' to exit): ")
  bayes_estimate_wrapper(distribution)
  
  if(distribution=="quit")
    break;
}

