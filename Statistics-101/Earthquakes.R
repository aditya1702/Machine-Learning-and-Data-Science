library(tidyverse)

shanghai <- read_csv('/Users/adityavyas/Desktop/Sem-1/Probability and Statistics/shanghai.csv')
shanghai <- shanghai[-1, ]

shanghai %>% 
filter(mag.vec1 > 3.05) %>%
ggplot(aes(mag.vec1))+
geom_density()

shanghai_vec1_earthquakes_greater_than_3_df <- shanghai %>% filter(mag.vec1 > 3.05)
sum(shanghai_vec1_earthquakes_greater_than_3_df)
v1a <- shanghai_vec1_earthquakes_greater_than_3_df$mag.vec1

v1b <- v1a - min(v1a)
v1b <- v1b + 1e-5

plot(density(v1b))

mom_normal(v1b)
mom_gamma(v1b)
mom_exponential(v1b)
mle_gamma(v1b)
parametric_bootstrap_using_ks_test(input_data = v1b, term = 'rgamma', distribution = 'gamma', distribution_function = rgamma, mle_function = mle_gamma, val = 2.32883, val2 = 0.324350)

mle_normal(v1b)
mle_gamma(v1b)
mle_exponential(v1b)
