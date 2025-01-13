install.packages("pacman")
pacman::p_load(R2jags, parallel, ggpubr, extraDistr, truncnorm)

set.seed(2000)

setwd('/work/IdaHeleneDencker#2808/Exam/Code/Parameter_recovery')

# defining a function for calculating the maximum of the posterior density (not exactly the same as the mode)
MPD <- function(x) {
  density(x)$x[which(density(x)$y==max(density(x)$y))]
}

#------ create task environment -------------------
# NB! mod(ntrials, nstruct) (aka. ntrials %% nstruct) must be 0
ntrials <- 100 # total number of trials in our payoff structure
nstruct <- 10 # size of our subdivisions for pseudorandomization
freq <- 0.5 # probability of our frequent losses (we have losses half of the time)
infreq <- 0.1 # probability of our infrequent losses (we have losses 1/10th of the time)
bad_r <- 100 # "bad" winnings
bad_freq_l <- -250 # "bad" frequent loss
bad_infreq_l <- -1250 # "bad" infrequent loss
good_r <- 50 # "good" winnings
good_freq_l <- -50 # "good" frequent loss
good_infreq_l <- -250 # "good" infrequent loss

#(R for reward, L for loose)
# Bad frequent
A_R <- rep(bad_r, nstruct) # we win on every trials
A_L <- c(rep(bad_freq_l, nstruct*freq),rep(0,nstruct*(1-freq))) # we have losses half of the time

# Bad infrequent
B_R <- rep(bad_r, nstruct)
B_L <- c(rep(bad_infreq_l, nstruct*infreq),rep(0,nstruct*(1-infreq))) # we have losses 1/10th of the time

# Good frequent
C_R <- rep(good_r, nstruct)
C_L <- c(rep(good_freq_l, nstruct*freq),rep(0,nstruct*(1-freq)))

# Good infrequent
D_R <- rep(good_r, nstruct)
D_L <- c(rep(good_infreq_l, nstruct*infreq),rep(0,nstruct*(1-infreq)))

# create the pseudorandomized full payoff structure
A <- array(NA,ntrials) # setting up an empty array to be filled
B <- array(NA,ntrials)
C <- array(NA,ntrials)
D <- array(NA,ntrials)
for (i in 1:(ntrials/nstruct)) {
  A[(1+(i-1)*nstruct):(i*nstruct)] <- (A_R + sample(A_L)) # randomly shuffling the loss-array for every ten trials (and adding those losses to the winnings)
  B[(1+(i-1)*nstruct):(i*nstruct)] <- (B_R + sample(B_L))
  C[(1+(i-1)*nstruct):(i*nstruct)] <- (C_R + sample(C_L))
  D[(1+(i-1)*nstruct):(i*nstruct)] <- (D_R + sample(D_L))
}

# "less generic" code
# A <- c()
# B <- c()
# C <- c()
# D <- c()
# for (i in 1:10) {
#   A <- (append(A,A_R + sample(A_L)))
#   B <- (append(B,B_R + sample(B_L)))
#   C <- (append(C,C_R + sample(C_L)))
#   D <- (append(D,D_R + sample(D_L)))
# }

payoff <- cbind(A,B,C,D)/100 # combining all four decks as columns with each 100 trials - dividing our payoffs by 100 to make the numbers a bit easier to work with

# let's look at the payoff
colSums(payoff) # the two bad decks should sum to -25 (i.e. -2500), and the two good ones to 25 (i.e. 2500)

###--------------Run full parameter recovery -------------
niterations <- 40 # how many times it runs through the model
nsubs <- 10 #number of subjects (participants) 
ntrials_all <- rep(100, 10) # all simulated subs have 100 trials each

# setting up empty arrays to be filled
# mu
true_mu_a_rew <- array(NA,c(niterations))
true_mu_a_pun <- array(NA,c(niterations))
true_mu_K <- array(NA,c(niterations))
true_mu_omega_f <- array(NA,c(niterations))
true_mu_omega_p <- array(NA,c(niterations))

infer_mu_a_rew <- array(NA,c(niterations))
infer_mu_a_pun <- array(NA,c(niterations))
infer_mu_K <- array(NA,c(niterations))
infer_mu_omega_f <- array(NA,c(niterations))
infer_mu_omega_p <- array(NA,c(niterations))

# sigma (SD for R) / lambda (precision for JAGS)
true_lambda_a_rew <- array(NA,c(niterations))
true_lambda_a_pun <- array(NA,c(niterations))
true_lambda_K <- array(NA,c(niterations))
true_lambda_omega_f <- array(NA,c(niterations))
true_lambda_omega_p <- array(NA,c(niterations))

infer_lambda_a_rew <- array(NA,c(niterations))
infer_lambda_a_pun <- array(NA,c(niterations))
infer_lambda_K <- array(NA,c(niterations))
infer_lambda_omega_f <- array(NA,c(niterations))
infer_lambda_omega_p <- array(NA,c(niterations))

print(paste("Running", niterations, "iterations for", nsubs, "subjects"))
start_time = Sys.time()
for (i in 1:niterations) {
  
  ntrials <- ntrials_all
  # In each iteration, the code generates random "true" parameter values for the hierarchical model representing ground-truth values
  # drawn using runif() (uniform distribution)
  
  #mu (means)
  mu_a_rew <- runif(1,0,1) #draw 1 sample between 0 and 1
  mu_a_pun <- runif(1,0,1) 
  mu_K <- runif(1,0,2) #draw 1 sample between 0 and 2
  mu_omega_f <- runif(1,-2,2)
  mu_omega_p <- runif(1,-2,2)
  
  #sigma (sd's)
  sigma_a_rew <- runif(1,0,0.1)
  sigma_a_pun <- runif(1,0,0.1)
  sigma_K <- runif(1,0,0.2)
  sigma_omega_f <- runif(1,0,0.4)
  sigma_omega_p <- runif(1,0,0.4)
  
  # sigma_a_rew <- runif(1,0,.5)
  # sigma_a_pun <- runif(1,0,.5)
  # sigma_K <- runif(1,0,.5)
  # sigma_theta <- runif(1,0,.5)
  # sigma_omega_f <- runif(1,0,.5)
  # sigma_omega_p <- runif(1,0,.5)
  
  source('ORL_hier_sim.R')
  ORL_sims <- hier_ORL_sim(payoff,nsubs,ntrials,mu_a_rew,mu_a_pun,
                           mu_K,mu_omega_f,mu_omega_p,
                           sigma_a_rew,sigma_a_pun,sigma_K,
                           sigma_omega_f,sigma_omega_p)
  
  x <- ORL_sims$x #get the choices (decks) from the simulations
  X <- ORL_sims$X #get the outcomes (payoffs) from the simulations
  
  # set up jags and run jags model
  data <- list("x","X","ntrials","nsubs") 
  
  params<-c("mu_a_rew","mu_a_pun",
            "mu_K","mu_omega_f","mu_omega_p","lambda_a_rew",
            "lambda_a_pun","lambda_K","lambda_omega_f","lambda_omega_p")
  
  samples <- jags.parallel(data, inits=NULL, params,
                           model.file ="ORL_hier.txt", n.chains=3, #should have at least 3 chains
                           n.iter=5000, n.burnin=1000, n.thin=1, n.cluster=4)
  
  
  
  # store True and Inferred Parameter Values
  # mu
  true_mu_a_rew[i] <- mu_a_rew
  true_mu_a_pun[i] <- mu_a_pun
  true_mu_K[i] <- mu_K
  true_mu_omega_f[i] <- mu_omega_f
  true_mu_omega_p[i] <- mu_omega_p
  
  # find maximum a posteriori
  Y <- samples$BUGSoutput$sims.list #The posterior samples for each parameter are extracted
  infer_mu_a_rew[i] <- MPD(Y$mu_a_rew) #The MAP estimate is calculated for each parameter using a custom MPD function
  infer_mu_a_pun[i] <- MPD(Y$mu_a_pun)
  infer_mu_K[i] <- MPD(Y$mu_K)
  infer_mu_omega_f[i] <- MPD(Y$mu_omega_f)
  infer_mu_omega_p[i] <- MPD(Y$mu_omega_p)
  
  # lambda
  true_lambda_a_rew[i] <- sigma_a_rew
  true_lambda_a_pun[i] <- sigma_a_pun
  true_lambda_K[i] <- sigma_K
  true_lambda_omega_f[i] <- sigma_omega_f
  true_lambda_omega_p[i] <- sigma_omega_p
  
  # find maximum a posteriori
  infer_lambda_a_rew[i] <- MPD(Y$lambda_a_rew) #The MAP estimate is calculated for each parameter using a custom MPD function
  infer_lambda_a_pun[i] <- MPD(Y$lambda_a_pun)
  infer_lambda_K[i] <- MPD(Y$lambda_K)
  infer_lambda_omega_f[i] <- MPD(Y$lambda_omega_f)
  infer_lambda_omega_p[i] <- MPD(Y$lambda_omega_p)
  
  print(i)
  
}

end_time = Sys.time()
end_time - start_time


#save
filename <- paste0("JAGS_hier_ORL_recovery_", niterations, "_", nsubs, ".RData")
save(ntrials_all, 
     niterations, 
     samples,
     infer_mu_a_pun,
     infer_mu_a_rew,
     infer_mu_K,
     infer_mu_omega_f,
     infer_mu_omega_p,
     true_mu_a_pun,
     true_mu_a_rew,
     true_mu_K,
     true_mu_omega_f,
     true_mu_omega_p,
     infer_lambda_a_pun,
     infer_lambda_a_rew,
     infer_lambda_K,
     infer_lambda_omega_f,
     infer_lambda_omega_p,
     true_lambda_a_pun,
     true_lambda_a_rew,
     true_lambda_K,
     true_lambda_omega_f,
     true_lambda_omega_p,
     file=filename)
print('saving recovery data')

#loading in data 
#load("JAGS_hier_ORL_recovery_no_theta.RData")

#look at model diagnostics and convergence
print(samples)
#Save 
filename <- paste0("jags_model_report_", niterations, "_", nsubs, ".txt")
capture.output(print(samples), file = filename)

# Plot traceplots for all the parameters
filename <- paste0("traceplots_parameters_", niterations, "_", nsubs, ".png")
png(filename, 
    width = 2000, 
    height = 2500,  
    res = 150)

# Adjust the margins for better aesthetics (increase space around plots)
par(mar = c(4, 4, 2, 1))  # Bottom, Left, Top, Right margins (increased to give space)

# Generate the trace plots
trace <- traceplot(samples, 
                   mfrow = c(4, 3),  
                   varname = c("deviance", "lambda_K", "lambda_a_pun", 
                               "lambda_a_rew", "lambda_omega_f", "lambda_omega_p",
                               "mu_K", "mu_a_pun", "mu_a_rew", 
                               "mu_omega_f", "mu_omega_p"), #lambda_theta, mu_theta
                   match.head = TRUE, 
                   ask = FALSE) 

# Close and save
dev.off()


# look at scatter plots
# designed to assess how well model recovers the "true" parameter values (true_*) from the simulated data. 
source('recov_plot.R') # Plotting code from Lasse

# 1 red dot is 1 itterration through the model i.e. Each point corresponds to a single simulated dataset and the model's attempt to recover the parameters.
# Dashed black Line (the Identity Line) is the reference line where true = inferred. Deviations from this line indicate inaccuracies in recovery. As we can see lambdas are far from the lines and mu's closer
# The Smoothed Linear Fit (Red Line) shows the overall relationship between true and inferred values using a linear regression.The gray region around the smoothed line shows the 95% confidence interval for the regression. A wide confidence interval (big gray area) indicates variability

#mu's
pl1 <- recov_plot(true_mu_a_rew, infer_mu_a_rew, c("true mu_a_rew", "infer mu_a_rew"), 'smoothed linear fit') +
  theme(plot.title = element_text(size = 10)) 
pl2 <- recov_plot(true_mu_a_pun, infer_mu_a_pun, c("true mu_a_pun", "infer mu_a_pun"), 'smoothed linear fit') +
  theme(plot.title = element_text(size = 10))
pl3 <- recov_plot(true_mu_K, infer_mu_K, c("true mu_K", "infer mu_K"), 'smoothed linear fit') +
  theme(plot.title = element_text(size = 10))
pl4 <- recov_plot(true_mu_omega_f, infer_mu_omega_f, c("true mu_omega_f", "infer mu_omega_f"), 'smoothed linear fit') +
  theme(plot.title = element_text(size = 10))
pl5 <- recov_plot(true_mu_omega_p, infer_mu_omega_p, c("true mu_omega_p", "infer mu_omega_p"), 'smoothed linear fit') +
  theme(plot.title = element_text(size = 10))
combined_mu <-  ggarrange(pl1, pl2, pl3, pl4, pl5, 
          labels = NULL, 
          font.label = list(size = 10)) # Adjust label font size
filename <- paste0("arranged_plot_mu_", niterations, "_", nsubs, ".png")
ggsave(filename, plot = combined_mu, width = 12, height = 8, dpi = 300)


#lambda's
pl6 <- recov_plot(true_lambda_a_rew, infer_lambda_a_rew, c("true lambda_a_rew", "infer lambda_a_rew"), 'smoothed linear fit') +
  theme(plot.title = element_text(size = 10))  
pl7 <- recov_plot(true_lambda_a_pun, infer_lambda_a_pun, c("true lambda_a_pun", "infer lambda_a_pun"), 'smoothed linear fit') +
  theme(plot.title = element_text(size = 10))
pl8 <- recov_plot(true_lambda_K, infer_lambda_K, c("true lambda_K", "infer lambda_K"), 'smoothed linear fit') +
  theme(plot.title = element_text(size = 10))
pl9 <- recov_plot(true_lambda_omega_f, infer_lambda_omega_f, c("true lambda_omega_f", "infer lambda_omega_f"), 'smoothed linear fit') +
  theme(plot.title = element_text(size = 10))
pl10 <- recov_plot(true_lambda_omega_p, infer_lambda_omega_p, c("true lambda_omega_p", "infer lambda_omega_p"), 'smoothed linear fit') +
  theme(plot.title = element_text(size = 10))
combined_lambda <-  ggarrange(pl6, pl7, pl8, pl9, pl10, 
                          labels = NULL, 
                          font.label = list(size = 10)) # Adjust label font size
filename <- paste0("arranged_plot_lambda_", niterations, "_", nsubs, ".png")
ggsave(filename, plot = combined_lambda, width = 12, height = 8, dpi = 300)

print('done running parameter recovery')
