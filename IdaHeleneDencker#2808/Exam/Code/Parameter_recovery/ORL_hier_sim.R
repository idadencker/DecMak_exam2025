# Function to simulate data for a hierarchical ORL model

# Parameters:
# payoff: Matrix of payoffs for each deck (rows = trials, columns = decks)
# nsubs: Number of subjects
# ntrials: Number of trials per subject (vector)
# mu_*: Group-level means for free parameters (e.g., mu_a_rew = mean reward sensitivity)
# sigma_*: Group-level standard deviations for free parameters (e.g., sigma_a_rew = SD of reward sensitivity)

hier_ORL_sim <- function(payoff, nsubs, ntrials, mu_a_rew, mu_a_pun,
                         mu_K, mu_omega_f, mu_omega_p,
                         sigma_a_rew, sigma_a_pun, sigma_K,
                         sigma_omega_f, sigma_omega_p) {
  
  # Initialize arrays for simulation outputs
  x <- array(NA, c(nsubs, ntrials[1])) # Choices made by subjects
  X <- array(NA, c(nsubs, ntrials[1])) # Payoffs received by subjects
  Ev <- array(NA, c(nsubs, ntrials[1], 4)) # Expected value for each deck
  
  # Loop over subjects to simulate individual-level data
  for (s in 1:nsubs) {
    
    # Sample free parameters for the subject from truncated normal distributions
    a_rew <- rtruncnorm(1, 0, , mu_a_rew, sigma_a_rew)       # Reward sensitivity
    a_pun <- rtruncnorm(1, 0, , mu_a_pun, sigma_a_pun)       # Punishment sensitivity
    K <- rtruncnorm(1, 0, , mu_K, sigma_K)                   # Decay rate for perseveration
    #theta <- rtruncnorm(1, 0, , mu_theta, sigma_theta)       # Inverse temperature (decision noise)
    theta <- 1 # fixed, to match literature 
    omega_f <- rtruncnorm(1, 0, , mu_omega_f, sigma_omega_f) # Weight for frequency of outcomes
    omega_p <- rtruncnorm(1, 0, , mu_omega_p, sigma_omega_p) # Weight for perseveration
    
    # Initialize arrays for trial-level updates
    Ev_update <- array(NA, c(ntrials[s], 4)) # Updates to expected value
    signX <- array(NA, c(ntrials[s]))        # Sign of the payoff
    Ef_cho <- array(NA, c(ntrials[s], 4))   # Expected frequency if a deck is chosen
    Ef_not <- array(NA, c(ntrials[s], 4))   # Expected frequency if a deck is not chosen
    Ef <- array(NA, c(ntrials[s], 4))       # Final expected frequencies
    PS <- array(NA, c(ntrials[s], 4))       # Perseveration scores
    V <- array(NA, c(ntrials[s], 4))        # Total value for each deck
    exp_p <- array(NA, c(ntrials[s], 4))    # Exponentiated values for softmax calculation
    p <- array(NA, c(ntrials[s], 4))        # Choice probabilities for each deck
    
    # Initialize first trial
    x[s, 1] <- rcat(1, c(0.25, 0.25, 0.25, 0.25)) # Random initial choice
    X[s, 1] <- payoff[1, x[s, 1]]                # Payoff for the first choice
    Ev[s, 1, ] <- rep(0, 4)                      # Initial expected values for all decks
    Ef[1, ] <- rep(0, 4)                         # Initial expected frequencies for all decks
    PS[1, ] <- rep(0, 4)                         # Initial perseveration scores
    
    # Loop over trials to simulate decisions and updates
    for (t in 2:ntrials) {
      
      # Determine the sign of the payoff (positive or negative)
      signX[t] <- ifelse(X[s, t-1] < 0, -1, 1)
      
      # Update values for each deck
      for (d in 1:4) {
        
        # Update expected values based on whether the payoff was a reward or punishment
        Ev_update[t, d] <- ifelse(X[s, t-1] >= 0,
                                  Ev[s, t-1, d] + a_rew * ((X[s, t-1]) - Ev[s, t-1, d]),
                                  Ev[s, t-1, d] + a_pun * ((X[s, t-1]) - Ev[s, t-1, d]))
        Ev[s, t, d] <- ifelse(d == x[s, t-1], Ev_update[t, d], Ev[s, t-1, d])
        
        # Update expected frequencies for chosen and unchosen decks
        Ef_cho[t, d] <- ifelse(X[s, t-1] >= 0,
                               Ef[t-1, d] + a_rew * (signX[t] - Ef[t-1, d]),
                               Ef[t-1, d] + a_pun * (signX[t] - Ef[t-1, d]))
        Ef_not[t, d] <- ifelse(X[s, t-1] >= 0,
                               Ef[t-1, d] + a_pun * (-(signX[t] / 3) - Ef[t-1, d]),
                               Ef[t-1, d] + a_rew * (-(signX[t] / 3) - Ef[t-1, d]))
        Ef[t, d] <- ifelse(d == x[s, t-1], Ef_cho[t, d], Ef_not[t, d])
        
        # Update perseveration scores
        PS[t, d] <- ifelse(x[s, t-1] == d, 1 / (1 + K), PS[t-1, d] / (1 + K))
        
        # Calculate total value for each deck
        V[t, d] <- Ev[s, t, d] + Ef[t, d] * omega_f + PS[t, d] * omega_p
        
        # Calculate exponentiated values for the softmax function
        exp_p[t, d] <- exp(theta * V[t, d])
      }
      
      # Calculate choice probabilities using the softmax function
      for (d in 1:4) {
        p[t, d] <- exp_p[t, d] / sum(exp_p[t, ])
      }
      
      # Make a choice for the current trial based on probabilities
      x[s, t] <- rcat(1, p[t, ])
      
      # Get the payoff for the chosen deck
      X[s, t] <- payoff[t, x[s, t]]
    }
  }
  
  # Return the results as a list
  result <- list(x = x,
                 X = X,
                 Ev = Ev)
  return(result)
}
