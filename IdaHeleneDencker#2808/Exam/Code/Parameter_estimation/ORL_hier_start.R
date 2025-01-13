install.packages("pacman")
pacman::p_load(R2jags, parallel,ggplot2,gridExtra)

#dev.off()
set.seed(2000)

setwd('/work/IdaHeleneDencker#2808/Exam/Code/Parameter_estimation')

# defining a function for calculating the maximum of the posterior density (not exactly the same as the mode)
MPD <- function(x) {
  density(x)$x[which(density(x)$y==max(density(x)$y))]
}

#load control data
ctr_data <- read.table("/work/IdaHeleneDencker#2808/Exam/Data/617_data/stein.txt",header=TRUE)
# The starting amount group

#----------prepare data for jags models - want trial x subject arrays for choice and gain & loss ----
# identify and count unique subject IDs
subIDs <- unique(ctr_data$subjID)
nsubs <- length(subIDs)
ntrials_max <- 100

# all choices (x) and outcomes (X)
x_raw <- ctr_data$deck
X_raw <- ctr_data$gain + ctr_data$loss #note the sign!

#--- assign choices and outcomes in trial x sub matrix

# empty arrays to fill
ntrials_all <- array(0,c(nsubs))
x_all <- array(0,c(nsubs,ntrials_max)) #the choices of all subjects 
X_all <- array(0,c(nsubs,ntrials_max)) #the payoffs of all subjects 

for (s in 1:nsubs) {
  
  #record n trials for subject s
  ntrials_all[s] <- length(x_raw[ctr_data$subjID==subIDs[s]])
  
  # Assign trials directly to arrays
  x_all[s, 1:ntrials_all[s]] <- x_raw[ctr_data$subjID == subIDs[s]]
  X_all[s, 1:ntrials_all[s]] <- X_raw[ctr_data$subjID == subIDs[s]]
  
}

# Scaling the payoffs (cuz the learning parameter becomes less relevant for very large payoffs/losses)
X_all <- X_all/100

#----------testing our data curation by running JAGS on one subject

# Now we'll fit one subject just to make sure everything works

x <- x_all[1,] #take only first participant
X <- X_all[1,]

ntrials <- ntrials_all[1]

print('fitting model on 1 (start group)')
# set up jags and run jags model on one subject
data <- list("x","X","ntrials") 
params<-c("a_rew","a_pun","K","omega_f","omega_p","p")
samples <- jags.parallel(data, inits=NULL, params,
                         model.file ="ORL.txt",
                         n.chains=2, n.iter=5000, n.burnin=1000, n.thin=1, n.cluster=4) 

#look at model diagnostics and convergence
print(samples)
#Save 
capture.output(print(samples), file = "start_jags_model_report_1_part.txt")


# Plot traceplots for all the parameters
png("start_traceplots_parameters_1_part.png", 
    width = 1000, 
    height = 1000,  
    res = 150)
# Adjust the margins for better aesthetics (increase space around plots)
par(mar = c(4, 4, 2, 1))  # Bottom, Left, Top, Right margins (increased to give space)
# Generate the trace plots
trace <- traceplot(samples, 
                   mfrow = c(3, 2),  
                   varname = c("a_rew","a_pun","K","omega_f","omega_p"),
                   match.head = TRUE, 
                   ask = FALSE) 
# Close and save
dev.off()


# let's look at the posteriors for the parameters
# Create individual ggplot density plots for each
plot_a_rew <- ggplot(data.frame(x = samples$BUGSoutput$sims.list$a_rew), aes(x)) + 
  geom_density() + 
  ggtitle("Density of a_rew")
plot_a_pun <- ggplot(data.frame(x = samples$BUGSoutput$sims.list$a_pun), aes(x)) + 
  geom_density() + 
  ggtitle("Density of a_pun")
plot_K <- ggplot(data.frame(x = samples$BUGSoutput$sims.list$K), aes(x)) + 
  geom_density() + 
  ggtitle("Density of K")
plot_omega_f <- ggplot(data.frame(x = samples$BUGSoutput$sims.list$omega_f), aes(x)) + 
  geom_density() + 
  ggtitle("Density of omega_f")
plot_omega_p <- ggplot(data.frame(x = samples$BUGSoutput$sims.list$omega_p), aes(x)) + 
  geom_density() + 
  ggtitle("Density of omega_p")

# Arrange all plots in a 3x2 grid
one_part <-grid.arrange(plot_a_rew, plot_a_pun, plot_K, plot_omega_f, plot_omega_p, ncol = 2)
ggsave('/work/IdaHeleneDencker#2808/Exam/Code/Parameter_estimation/start_posteriors_parameters_1_part.png', plot = one_part, width = 12, height = 8, dpi = 300)



###########################################################
#---------- run the hierarchical model on controls --------
###########################################################

x <- x_all # now take all participants
X <- X_all

ntrials <- ntrials_all

# set up jags and run jags model
data <- list("x","X","ntrials","nsubs") 
# NB! we're not tracking theta cuz we're not modelling it in order reduce complexity a bit (hence, we're just setting it to 1)
params<-c("mu_a_rew","mu_a_pun","mu_K","mu_omega_f","mu_omega_p") 

print('fitting model on all (start group)')
samples <- jags.parallel(data, inits=NULL, params,
                         model.file ="ORL_hier.txt",
                         n.chains=3, n.iter=5000, n.burnin=1000, n.thin=1, n.cluster=4)

#save
save(samples,
     file="start_JAGS_ORL_hier.RData")
print('saving samples')

#loading in data 
#load("JAGS_ORL_hier_start.RData")

#look at model diagnostics and convergence
print(samples)
#Save 
capture.output(print(samples), file = "start_jags_model_report_all_part.txt")
#dev.off()


# Plot traceplots for all the parameters
png("start_traceplots_parameters_all_part.png", 
    width = 1000, 
    height = 1000,  
    res = 150)

# Adjust the margins for better aesthetics (increase space around plots)
par(mar = c(4, 4, 2, 1))  # Bottom, Left, Top, Right margins (increased to give space)

# Generate the trace plots
trace <- traceplot(samples, 
                   mfrow = c(3, 2),  
                   varname = c("mu_a_rew","mu_a_pun","mu_K","mu_omega_f","mu_omega_p"),
                   match.head = TRUE, 
                   ask = FALSE) 
# Close and save
dev.off()


# Create individual ggplot density plots
plot_a_rew <- ggplot(data.frame(x = samples$BUGSoutput$sims.list$mu_a_rew), aes(x)) + 
  geom_density() + 
  ggtitle("Density of mu_a_rew")+
  xlim(c(0, 1)) +
  ylim(c(0, 7))
plot_a_pun <- ggplot(data.frame(x = samples$BUGSoutput$sims.list$mu_a_pun), aes(x)) + 
  geom_density() + 
  ggtitle("Density of mu_a_pun")+
  xlim(c(0, 0.3)) +
  ylim(c(0, 33))
plot_K <- ggplot(data.frame(x = samples$BUGSoutput$sims.list$mu_K), aes(x)) + 
  geom_density() + 
  ggtitle("Density of mu_K")+
  xlim(c(0, 6)) +
  ylim(c(0, 0.8))
plot_omega_f <- ggplot(data.frame(x = samples$BUGSoutput$sims.list$mu_omega_f), aes(x)) + 
  geom_density() + 
  ggtitle("Density of mu_omega_f")+
  xlim(c(-5, 5)) +
  ylim(c(0, 1.6))
plot_omega_p <- ggplot(data.frame(x = samples$BUGSoutput$sims.list$mu_omega_p), aes(x)) + 
  geom_density() + 
  ggtitle("Density of mu_omega_p")+
  xlim(c(-12, 12)) +
  ylim(c(0, 0.5))

# Arrange all plots in a 3x2 grid
all_part <- grid.arrange(plot_a_rew, plot_a_pun, plot_K, plot_omega_f, plot_omega_p, ncol = 2)
ggsave('/work/IdaHeleneDencker#2808/Exam/Code/Parameter_estimation/start_posteriors_parameters_all_part.png', plot = all_part, width = 12, height = 8, dpi = 300)

print("done running parameter estimation start")