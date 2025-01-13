install.packages("pacman")
pacman::p_load(R2jags, parallel, polspline, ggplot2,gridExtra,logspline)

set.seed(2000)

setwd('/work/IdaHeleneDencker#2808/Exam/Code/Group_comparison')

# defining a function for calculating the maximum of the posterior density (not exactly the same as the mode)
MPD <- function(x) {
  density(x)$x[which(density(x)$y==max(density(x)$y))]
}

#load control data
stein_data <- read.table("/work/IdaHeleneDencker#2808/Exam/Data/617_data/stein.txt",header=TRUE)
own_data <- read.table("/work/IdaHeleneDencker#2808/Exam/Data/Own_data/own.txt",header=TRUE)

#----------prepare data for jags models - want trial x subject arrays for choice, gain, and loss ----

# identify and count unique subject IDs
subIDs_stein <- unique(stein_data$subjID)
nsubs_stein <- length(subIDs_stein)

subIDs_own <- unique(own_data$subjID)
nsubs_own <- length(subIDs_own)

ntrials_max <- 100

# SCALE THE REWARDS DOWN SO WE DON'T NEED SUCH WIDE PRIORS
# all choices (x) and outcomes (X) for both groups
# scale by 100
x_raw_stein <- stein_data$deck
X_raw_stein <- (stein_data$gain + stein_data$loss)/100 #note the sign!

x_raw_own <- own_data$deck
X_raw_own <- (own_data$gain + own_data$loss)/100 #note the sign!


#--- assign choices and outcomes in trial x sub matrix
# empty arrays to fill
ntrials_stein <- array(0,c(nsubs_stein))
x_stein <- array(0,c(nsubs_stein,ntrials_max))
X_stein <- array(0,c(nsubs_stein,ntrials_max))

ntrials_own <- array(0,c(nsubs_own))
x_own <- array(0,c(nsubs_own,ntrials_max))
X_own <- array(0,c(nsubs_own,ntrials_max))


# make control data matrices
for (s in 1:nsubs_stein) {
  
  #record n trials for subject s
  ntrials_stein[s] <- length(x_raw_stein[stein_data$subjID==subIDs_stein[s]])
  
  # assign arrays
  x_stein[s, 1:ntrials_stein[s]] <- x_raw_stein[stein_data$subjID == subIDs_stein[s]]
  X_stein[s, 1:ntrials_stein[s]] <- X_raw_stein[stein_data$subjID == subIDs_stein[s]]
  
}

# make own data matrices
for (s in 1:nsubs_own) {
  
  #record n trials for subject s
  ntrials_own[s] <- length(x_raw_own[own_data$subjID==subIDs_own[s]])
  
  # assign arrays
  x_own[s, 1:ntrials_own[s]] <- x_raw_own[own_data$subjID == subIDs_own[s]]
  X_own[s, 1:ntrials_own[s]] <- X_raw_own[own_data$subjID == subIDs_own[s]]
  
}

data <- list("x_stein","X_stein","ntrials_stein","nsubs_stein",
             "x_own","X_own","ntrials_own","nsubs_own")

# 5 parameters
params<-c("alpha_a_rew","alpha_a_pun","alpha_K","alpha_omega_f","alpha_omega_p") 
#alpha_ parameters act as hyperparameters that influence how much the prior distribution for the model parameters (like a_rew_stein, a_pun_stein, K_stein, etc.) can shift. By adjusting these alpha_ values, you control the degree of flexibility
#e.g. alpha_a_rew parameter measures the difference in learning rate of reward for the 2 groups
# If alpha_a_rew is large, the model allows a larger difference in the reward learning rates for the two groups, making the two groups more distinct in terms of how they learn from rewards.

print('running JAGS for 5 alpha parameters')
start_time = Sys.time()
samples <- jags.parallel(data, inits=NULL, params,
                         model.file ="ORL_compare.txt",
                         n.chains=2, n.iter=5000, n.burnin=1000, n.thin=1, n.cluster=4) #3 chains will give an error 
end_time = Sys.time()
end_time - start_time


#save
save(samples,
     file="JAGS_ORL_hier_5_param.RData")
print('saving samples')

#loading in data 
#load("JAGS_ORL_hier_5_param.RData")

#look at model diagnostics and convergence
print(samples)
#Save 
capture.output(print(samples), file = "jags_model_report_5_param.txt")
#dev.off()

# Plot traceplots for all the parameters
png("traceplots_5_param.png", 
    width = 1000, 
    height = 1000,  
    res = 150)
# Adjust the margins for better aesthetics (increase space around plots)
par(mar = c(4, 4, 2, 1))  # Bottom, Left, Top, Right margins (increased to give space)
# Generate the trace plots
trace <- traceplot(samples, 
                   mfrow = c(3, 2),  
                   varname = c("alpha_a_rew","alpha_a_pun","alpha_K","alpha_omega_f","alpha_omega_p"),
                   match.head = TRUE, 
                   ask = FALSE) 
# Close and save
dev.off()

# calculating Bayes factors by comparing the posterior distribution of each parameter under the model to the prior distribution.
# The idea is to compute how the posterior evidence (the data) compares with the prior assumption about the parameter.

# create empty elements for storing the bayes factors
BF_effect <- NULL
BF_null <- NULL

fit.posterior <- logspline(samples$BUGSoutput$sims.list$alpha_a_rew) # fits a log-spline to the posterior samples of a parameter.
null.posterior <- dlogspline(0, fit.posterior) #Density of posterior at 0 = height of spline at 0:  calculates the value of the posterior density at 0 (the value under the null hypothesis of no effect). By doing this, you're evaluating how likely the null hypothesis is given the observed data.
null.prior     <- dnorm(0,0,(1/sqrt(1))) #Density of prior at 0 (remember to convert precision): calculates the density of the prior distribution at 0. using a normal prior distribution centered at 0 with a precision of 1 (standard deviation = 1). This represents the prior belief that there is no effect, with no evidence yet from the data.                  
BF_effect$alpha_a_rew <- null.prior/null.posterior #Bayes factor for the effect (alternative hypothesis). value greater than 1 indicates that the data provides more support for the alternative hypothesis than the prior.
BF_null$alpha_a_rew <- null.posterior/null.prior # Bayes factor for the null hypothesis, comparing how likely the data supports the null hypothesis (no effect) relative to the prior belief

fit.posterior <- logspline(samples$BUGSoutput$sims.list$alpha_a_pun)
null.posterior <- dlogspline(0, fit.posterior)
null.prior     <- dnorm(0,0,(1/sqrt(1)))                   
BF_effect$alpha_a_pun <- null.prior/null.posterior
BF_null$alpha_a_pun <- null.posterior/null.prior

fit.posterior <- logspline(samples$BUGSoutput$sims.list$alpha_omega_f)
null.posterior <- dlogspline(0, fit.posterior)
null.prior     <- dnorm(0,0,(1/sqrt(1)))                   
BF_effect$alpha_omega_f <- null.prior/null.posterior
BF_null$alpha_omega_f <- null.posterior/null.prior

fit.posterior <- logspline(samples$BUGSoutput$sims.list$alpha_omega_p)
null.posterior <- dlogspline(0, fit.posterior)
null.prior     <- dnorm(0,0,(1/sqrt(1)))                   
BF_effect$alpha_omega_p <- null.prior/null.posterior
BF_null$alpha_omega_p <- null.posterior/null.prior

fit.posterior <- logspline(samples$BUGSoutput$sims.list$alpha_K)
null.posterior <- dlogspline(0, fit.posterior)
null.prior     <- dnorm(0,0,(1/sqrt(1)))                   
BF_effect$alpha_K <- null.prior/null.posterior
BF_null$alpha_K <- null.posterior/null.prior

# Create data frames
BF_effect_df <- data.frame(
  Parameter = names(BF_effect),
  BayesFactor = unlist(BF_effect, use.names = FALSE))
BF_null_df <- data.frame(
  Parameter = names(BF_null),
  BayesFactor = unlist(BF_null, use.names = FALSE))
# Save as CSV files
write.csv(BF_effect_df, file = file.path('/work/IdaHeleneDencker#2808/Exam/Code/Group_comparison/BF_effect_5_param.csv'), row.names = FALSE)
write.csv(BF_null_df, file = file.path('/work/IdaHeleneDencker#2808/Exam/Code/Group_comparison/BF_null_5_param.csv'), row.names = FALSE)


## Create all plots
# 12000 samples, y-axis displaying 0-0.8
p1 <- ggplot() +
  geom_density(data = data.frame(x = rnorm(12000, 0, 1/sqrt(1))), aes(x = x), color = "black") +   # Prior beliefs about parameter (uninformed prior)
  geom_density(data = data.frame(x = samples$BUGSoutput$sims.list$alpha_a_rew), aes(x = x), color = "red") +  # Red line for param. (posterior)
  ylim(0, .9) +  # Set y-axis limits
  ggtitle("alpha_a_rew") 
theme_minimal() 

p2 <- ggplot() +
  geom_density(data = data.frame(x = rnorm(12000, 0, 1/sqrt(1))), aes(x = x), color = "black") +    # Prior beliefs about parameter (uninformed prior)
  geom_density(data = data.frame(x = samples$BUGSoutput$sims.list$alpha_a_pun), aes(x = x), color = "red") +  # Red line for param. (posterior)
  ylim(0, .9) +  # Set y-axis limits
  ggtitle("alpha_a_pun") 
theme_minimal()  

p3 <- ggplot() +
  geom_density(data = data.frame(x = rnorm(12000, 0, 1/sqrt(1))), aes(x = x), color = "black") +    # Prior beliefs about parameter (uninformed prior)
  geom_density(data = data.frame(x = samples$BUGSoutput$sims.list$alpha_omega_f), aes(x = x), color = "red") +  # Red line for param. (posterior)
  ylim(0, .9) +  # Set y-axis limits
  ggtitle("alpha_omega_f") 
theme_minimal()  

p4 <- ggplot() +
  geom_density(data = data.frame(x = rnorm(12000, 0, 1/sqrt(1))), aes(x = x), color = "black") +    # Prior beliefs about parameter (uninformed prior)
  geom_density(data = data.frame(x = samples$BUGSoutput$sims.list$alpha_omega_p), aes(x = x), color = "red") +  # Red line for param.(posterior)
  ylim(0, .9) +  # Set y-axis limits
  ggtitle("alpha_omega_p") 
theme_minimal()  

p5 <- ggplot() +
  geom_density(data = data.frame(x = rnorm(12000, 0, 1/sqrt(1))), aes(x = x), color = "black") +    # Prior beliefs about parameter (uninformed prior)
  geom_density(data = data.frame(x = samples$BUGSoutput$sims.list$alpha_K), aes(x = x), color = "red") +  # Red line for param.(posterior)
  ylim(0, .9) +  # Set y-axis limits
  ggtitle("alpha_K") 
theme_minimal()  
# Save the plots
all_plots <- grid.arrange(p1, p2, p3, p4, p5, ncol = 2) 
ggsave("/work/IdaHeleneDencker#2808/Exam/Code/Group_comparison/density_plot_5_parameters.png", plot = all_plots, width = 12, height = 8, dpi = 300)





############### Look at cumulative balance to justify learning ##############

# calculate cumulative value
xcum_stein <- array(0,c(nsubs_stein,100))
for (i in 1:nsubs_stein) {
  xcum_stein[i,] <- cumsum(X_stein[i,]) 
}

xcum_own <- array(0,c(nsubs_own,100))
for (i in 1:nsubs_own) {
  xcum_own[i,] <- cumsum(X_own[i,]) 
}


png("cumulative_scores.png", width = 1000, height = 1000, res = 150)
# Generate the trace plots
cum_plot <- plot(colMeans(xcum_stein), ylim = c(-15, 8), type = 'l', lwd = 2,  
                 xlab = "Trial", ylab = "Cumulative Balance") # should be -1500, 800 if not scaled
lines(colMeans(xcum_own), col = "red", lwd = 2)
# Add a legend
legend("topright",  
       legend = c("+2000", "+0"), 
       col = c("black", "red"),  # Colors corresponding to the lines
       lwd = 2, 
       cex = 0.8)  

# Close and save
dev.off()

############### compare final means ##############

# set up jags and run jags model on one subject
data <- list("xcum_stein","nsubs_stein",
             "xcum_own","nsubs_own") 
params<-c("mu","alpha","Smu_stein","Smu_own") # track individual means(Smu_stein, Smu_own), group mean (mu) and difference (alpha)

print('Running JAGS for means')
temp_samples <- jags.parallel(data, inits=NULL, params,
                              model.file ="balance_compare.txt",
                              n.chains=3, n.iter=15000, n.burnin=1000, n.thin=3, n.cluster=4)


#save
save(temp_samples,
     file="JAGS_ORL_hier_final_means.RData")
print('saving samples')

#loading in data 
#load("JAGS_ORL_hier_final_means.RData")

#look at model diagnostics and convergence
print(temp_samples)
#Save 
capture.output(print(temp_samples), file = "jags_model_report_final_means.txt")
#dev.off()

# Plot traceplots for all the parameters
png("traceplots_final_means.png", 
    width = 1000, 
    height = 600,  
    res = 150)
# Adjust the margins for better aesthetics (increase space around plots)
par(mar = c(4, 4, 2, 1))  # Bottom, Left, Top, Right margins (increased to give space)
# Generate the trace plots

trace <- traceplot(temp_samples, 
                   mfrow = c(1, 2),  
                   varname = c("mu","alpha","Smu_stein","Smu_own"),
                   match.head = TRUE, 
                   ask = FALSE) 
# Close and save
dev.off()


# savage dickey plot
#posterior for the difference 
p_alpha <- ggplot() +
  geom_density(data = data.frame(x = rnorm(12000, 0, 1/sqrt(1))), aes(x = x), color = "black") +    # Prior beliefs about parameter (uninformed prior)
  geom_density(data = data.frame(x = temp_samples$BUGSoutput$sims.list$alpha), aes(x = x), color = "red") +  # Red line for param. (posterior)
  ylim(0, .5) +  # Set y-axis limits
  ggtitle("Final means alpha") 
theme_minimal()  

p_mu <- ggplot() +
  geom_density(data = data.frame(x = rnorm(12000, 0, 1/sqrt(1))), aes(x = x), color = "black") +    # Prior beliefs about parameter (uninformed prior)
  geom_density(data = data.frame(x = temp_samples$BUGSoutput$sims.list$mu), aes(x = x), color = "red") +  # Red line for param.(posterior)
  ylim(0, .5) +  # Set y-axis limits
  ggtitle("Final means mu") 
theme_minimal() 
# Save the plots
two_plots <- grid.arrange(p_alpha,p_mu, ncol = 2) 
ggsave("/work/IdaHeleneDencker#2808/Exam/Code/Group_comparison/density_plot_final_means.png", plot = two_plots, width = 12, height = 8, dpi = 300)


#Bayes factor for effect and null hypothesis (for mu (overall mean) and alpha (overall difference) of accumulative score)
fit.posterior <- logspline(temp_samples$BUGSoutput$sims.list$alpha)
null.posterior <- dlogspline(0, fit.posterior)
null.prior     <- dnorm(0,0,(1/sqrt(1)))                   
BF_effect_alpha <- null.prior/null.posterior
BF_null_alpha <- null.posterior/null.prior

fit.posterior <- logspline(temp_samples$BUGSoutput$sims.list$mu)
null.posterior <- dlogspline(0, fit.posterior)
null.prior     <- dnorm(0,0,(1/sqrt(1)))                   
BF_effect_mu <- null.prior/null.posterior
BF_null_mu <- null.posterior/null.prior


#make and save df of BF
bayes_factor_df_effect <- data.frame(
  Parameter = c("Alpha", "Mu"),
  BF_Effect = c(BF_effect_alpha, BF_effect_mu))
write.csv(bayes_factor_df_effect,'/work/IdaHeleneDencker#2808/Exam/Code/Group_comparison/BF_effect_final_means.csv', row.names = FALSE)

bayes_factor_df_null <- data.frame(
  Parameter = c("Alpha", "Mu"),
  BF_Effect = c(BF_null_alpha, BF_null_mu))
write.csv(bayes_factor_df_null,'/work/IdaHeleneDencker#2808/Exam/Code/Group_comparison/BF_null_final_means.csv', row.names = FALSE)


print('done running group comparison')