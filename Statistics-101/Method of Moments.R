# METHOD OF MOMENT ESTIMATORS ASSIGNMENT
# CREATED BY - Vedant Choudhary, Aditya Vyas
library(Rlab)

############################################################################
########################## UTILITY FUNCTIONS ###############################
############################################################################
# The basic idea behind this form of the method is to:
#   (1) Equate the first sample moment about the origin M1=1n∑i=1nXi=X¯ to the first theoretical moment E(X).
#   (2) Equate the second sample moment about the mean M∗2=1n∑i=1n(Xi−X¯)2 to the second theoretical moment about the mean E[(X−μ)2].
# Reference for the above lines: https://onlinecourses.science.psu.edu/stat414/node/193/

# First moment has been calculated about the origin
firstMomentCalc <- function(input_data){
  temp = 0
  for (i in input_data){
    temp = temp + i
  }
  return (temp/length(input_data))
}

# Second moment has been calculated about the mean
secondMomentCalc <- function(input_data, x_bar){
  temp = 0
  for (i in input_data){
    temp = temp + (i - x_bar)^2
  }
  return (temp/length(input_data))
}

############################################################################
######### METHOD OF MOMENTS FUNCTIONS FOR DIFFERENT DISTRIBUTIONS ##########
############################################################################
# Point Distribution
mom_point <- function(input_data){
  # Estimating the parameters 
  a_hat = firstMomentCalc(input_data)
  print(paste("Estimated parameter 1 through MOM:", a_hat))
}

# Bernoulli Distribution
mom_bernoulli <- function(input_data){
  # Estimating the parameters 
  p_hat = firstMomentCalc(input_data)
  print(paste("Estimated parameter 1 through MOM:", p_hat))
}

# Binomial Distribution
mom_binomial <- function(input_data){
  mu_hat = mean(input_data)
  var_hat = secondMomentCalc(input_data, mu_hat)
  p_hat = (mu_hat - var_hat)/mu_hat
  n_hat = mu_hat^2/((mu_hat - var_hat))
  print(paste("Estimated parameter 1 through MOM:", p_hat))
  print(paste("Estimated parameter 2 through MOM:", n_hat))
}

# Geometric Distribution
mom_geometric <- function(input_data){ 
  p_hat = firstMomentCalc(input_data)
  print(paste("Estimated parameter 1 through MOM:", 1/p_hat))
}

# Poisson Distribution
mom_poisson <- function(input_data){ 
  lambda_hat = firstMomentCalc(input_data)
  print(paste("Estimated parameter 1 through MOM:", lambda_hat))
}

# Uniform Distribution
mom_uniform <- function(input_data){
  mu_hat = firstMomentCalc(input_data)
  var_hat = secondMomentCalc(input_data, mu_hat)
  a_hat = mu_hat - sqrt(3)*sqrt(var_hat)
  b_hat = mu_hat + sqrt(3)*sqrt(var_hat)
  print(paste("Estimated parameter 1 through MOM:", a_hat))
  print(paste("Estimated parameter 2 through MOM:", b_hat))
}

# Normal Distribution
mom_normal <- function(input_data){
  # Estimating the parameters
  mu_hat = firstMomentCalc(input_data)
  var_hat = secondMomentCalc(input_data, mu_hat)
  #print(paste("Estimated parameter 1 through MOM:", mu_hat))
  #print(paste("Estimated parameter 2 through MOM:", var_hat))
  return(c(mu_hat,var_hat))
}

# Exponential Distribution
mom_exponential <- function(input_data){ 
  theta_hat = firstMomentCalc(input_data)
  print(paste("Estimated parameter 1 through MOM:", 1/theta_hat))
}

# Gamma Distribution
mom_gamma <- function(input_data){
  mu_hat = firstMomentCalc(input_data)
  var_hat = secondMomentCalc(input_data, mu_hat)
  # We know that in gamma function, there are two parameters alpha and beta, with relation alpha*beta = mu and alpha*(beta^2) = variance
  theta_hat = var_hat/mu_hat
  alpha_hat = mu_hat/ theta_hat
  print(paste("Estimated parameter 1 through MOM:", alpha_hat))
  print(paste("Estimated parameter 2 through MOM:", theta_hat))
}

# Beta Distribution
mom_beta <- function(input_data){
  mu_hat = firstMomentCalc(input_data)
  var_hat = secondMomentCalc(input_data, mu_hat)
  beta_hat = (1-mu_hat)*((mu_hat*(1-mu_hat)/var_hat) - 1)
  alpha_hat = mu_hat*((mu_hat*(1-mu_hat)/var_hat) - 1)
  print(paste("Estimated parameter 1 through MOM:", alpha_hat))
  print(paste("Estimated parameter 2 through MOM:", beta_hat))
}

# T Distribution
mom_t <- function(input_data){
  mu_hat = firstMomentCalc(input_data)
  var_hat = secondMomentCalc(input_data, mu_hat)
  dof_hat = 2*var_hat/(var_hat-1)
  print(paste("Estimated parameter 1 through MOM:", dof_hat))
}

# Chi Square Distribution
mom_chisq <- function(input_data){
  p_hat = firstMomentCalc(input_data)
  print(paste("Estimated parameter 1 through MOM:", p_hat))
}

# Multinomial Distribution
mom_multinomial <- function(input_data){
  a = nrow(x)
  p = c(0,0,0,0,0)
  for(i in 1:a)
    p[i]<-1-((var(x[i,]))/mean(x[i,]))
  n = sum(rowMeans(x))/sum(p[1:a])
  print("For Multinomial Distribution the parameter n is")
  print(n)
  print("The parameter p is ")
  print(p)
}

# Multinormal Distribution
mom_multinormal <- function(input_data){
  mu_hat = colMeans(input_data)
  summation = var(input_data)
  print("For Multinormal Distribution the parameter mu_hat is")
  print(mu_hat)
  print("The parameter summation is ")
  print(summation)
}

############################################################################
########### MAIN CALL FUNCTION FOR METHOD OF MOMENTS FUNCTIONS #############
############################################################################
# Input - Distribution name
#         Population - If user does not want to insert a population, in-built populations will be used
#         User can also change in-built populations according to his/her need
mom_wrapper <- function(distribution, population = 0){
  if (distribution == "point"){
    input_data = population
    print("Point distribution has 1 parameter, hence 1st moment will give an estimator for a")
    estimator <- mom_point(input_data)
  }
  else if (distribution == "bernoulli"){
    if (population == 0){
      p = 0.5
      input_data = rbern(10000,p)  
    } else{
      input_data = population
    }
    print("Population parameter: ")
    print(p)
    # Sample data can be customized according to the user. 
    # As sample size goes towards population size, the estimators reach towards population parameter values
    sample_data = sample(input_data, 1000)
    print("Bernoulli distribution has 1 parameter, hence 1st moment will give an estimator for p")
    estimator <- mom_bernoulli(sample_data)
  }
  else if (distribution == "binomial"){
    if (population == 0){
      n = 100
      p = 0.5
      input_data = rbinom(10000,n,p)  
    } else{
      input_data = population
    }
    print("Population parameters: ")
    print(paste(p,",",n))
    sample_data = sample(input_data, 1000)
    print("Binomial distribution has 2 parameters - n and p - hence first two moments will give its parameter estimates")
    estimator <- mom_binomial(sample_data)
  }
  else if (distribution == "geometric"){
    if (population == 0){
      p = 0.5
      input_data = rgeom(10000,p)  
    } else{
      input_data = population
    }
    print("Population parameters: ")
    print(p)
    sample_data = sample(input_data, 1000)
    print("Geometric distribution has 1 parameter, hence 1st moment will give an estimator for p")
    estimator <- mom_geometric(sample_data)
  }
  else if (distribution == "poisson"){
    if (population == 0){
      lambda = 0.5
      input_data = rpois(10000,lambda)  
    } else{
      input_data = population
    }
    print("Population parameters: ")
    print(lambda)
    sample_data = sample(input_data, 1000)
    print("Poisson distribution has 1 parameter, hence 1st moment will give an estimator for lambda")
    estimator <- mom_poisson(input_data)
  }
  else if (distribution == "uniform"){
    if (population == 0){
      a = 0
      b = 100
      input_data = runif(10000,a,b)  
    } else{
      input_data = population
    }
    print("Population parameters: ")
    print(paste(a,",",b))
    print("Uniform distribution has 2 parameters - a and b - hence first two moments will give its parameter estimates")
    sample_data = sample(input_data, 1000)
    estimator <- mom_uniform(sample_data)
  }
  else if (distribution == "normal"){
    if (population == 0){
      input_data = rnorm(10000,0,1)  
    } else{
      input_data = population
    }
    m = mean(input_data)
    v = var(input_data)
    print("Population mean: ")
    print(m)
    print("Population variance: ")
    print(v)
    print("Normal distribution has 2 parameters - mu and sigma - hence first two moments will give its parameter estimates")
    sample_data = sample(input_data, length(input_data))
    estimator <- mom_normal(sample_data)
    # Here, two plots have been drawn to give an instructive visualization of how different the sample distribution is from
    # population distribution. The parameters for sample distribution are the ones estimated by method of moments
    # We have only plotted normal distribution to give an example of what else can be done with this assignment
    plot(input_data, dnorm(input_data, m, v), title("Population vs Sample"), col='red')
    points(sample_data, dnorm(sample_data, estimator[1], estimator[2]), col='green')
  }
  else if (distribution == "exponential"){
    if (population == 0){
      theta = 4
      input_data = rexp(10000,theta)  
    } else{
      input_data = population
    }
    print("Population parameter: ")
    print(theta)
    print("Exponential distribution has 1 parameter, hence 1st moment will give an estimator for theta")
    sample_data = sample(input_data, 1000)
    estimator <- mom_exponential(sample_data)
  }
  else if (distribution == "gamma"){
    if (population == 0){
      alpha = 4.7
      beta = 2.9
      input_data = rgamma(10000, scale = alpha, rate = beta)  
    } else{
      input_data = population
    }
    print("Population parameters: ")
    print(paste(alpha,",",beta))
    print("Gamma distribution has 2 parameters - alpha and theta - hence first two moments will give its parameter estimates")
    sample_data = sample(input_data, 1000)
    estimator <- mom_gamma(sample_data)
  }
  else if (distribution == "beta"){
    # Assuming alpha and beta are in the range [0,1] - WHAT WE HAVE BEEN TAUGHT IN CLASS
    # When the distribution is required over a known interval other than [0, 1] with random variable X, say [a, c] with random variable Y,
    # then replace x_bar can be replaced by (y_bar - a)/(c - a), and v_bar can be replaced by vy_bar/(c-a)^2
    if (population == 0){
      alpha = 5
      beta = 4
      input_data = rbeta(10000, shape1 = alpha, shape2 = beta)  
    } else{
      input_data = population
    }
    print("Population parameters: ")
    print(paste(alpha,",",beta))
    print("Beta distribution has 2 parameters - alpha and beta - hence first two moments will give its parameter estimates")
    sample_data = sample(input_data, 1000)
    estimator <- mom_beta(sample_data)
  }
  else if (distribution == "t"){
    # Assumg degree of freedom to be above 2, since parameters are only related to variance once d.o.g. is above 2 
    if (population == 0){
      dog = 5
      input_data = rt(10000, df = dog)  
    } else{
      input_data = population
    }
    print("Population parameter: ")
    print(dog)
    print("T distribution has 1 parameter, hence the 1st moment will give an estimator for v (degree of freedoms)")
    sample_data = sample(input_data, 1000)
    estimator <- mom_t(sample_data)
  }
  else if (distribution == "chi square"){
    if (population == 0){
      dog = 5
      input_data = rchisq(10000, df = dog)  
    } else{
      input_data = population
    }
    print("Population parameter: ")
    print(dog)
    print("Chi-Square distribution has 1 parameter, hence the 1st moment will give an estimator for p")
    sample_data = sample(input_data, 1000)
    estimator <- mom_chisq(sample_data)
  }
  else if (distribution == "multinomial"){
    if (population == 0){
      p = c(0.15,0.05,0.4,0.1,0.3)
      input_data = rmultinom(10000,size=5,p)
      a = nrow(input_data)
    } else{
      input_data = population
    }
    print("Population parameters: ")
    print(a)
    print(p)
    print("Multinomial distribution has following estimated parameters: ")
    sample_data = input_data
    estimator <- mom_multinomial(sample_data)
  }
  else if (distribution == "multinormal"){
    if (population == 0){
      vari = c(10,3,3,2)
      sigma = matrix(vari,2,2)
      input_data = mvrnorm(n = 1000, rep(0, 2), sigma)  
    } else{
      input_data = population
    }
    print("Population parameter: ")
    print(vari)
    print("Multinormal distribution has following estimated parameters")
    sample_data = input_data
    estimator <- mom_multinormal(sample_data)
  }
}

# STATEMENTS TO BE EXECUTED BEFORE RUNNING THE MAIN M.O.M. FUNCTION
# You can generate a population
population = sample(seq(1, 10000), 10000)

# Valid distributions for the program
repeat{
  cat("Valid Distributions: \n1. Point\n2. Bernoulli\n3. Binomial\n4. Geometric\n5. Poisson\n6. Uniform\n7. Normal\n8. Exponential\n9. Gamma\n10. Beta\n11. T\n12. Chi-Square\n13. Multinomial\n14. Multinormal")
  if_pop = readline(prompt = "Do you want to send a random population or use system's default populations (y/n): ")
  distribution <- readline(prompt="For which distribution, do you want M.O.M. estimation (write 'quit' to exit): ")
  if (tolower(if_pop) == "y"){
    mom_wrapper(distribution, population)
  }else{
    mom_wrapper(distribution)
  }
  if(distribution=="quit")
    break;
}
