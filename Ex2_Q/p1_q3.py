import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# All using vanilla Q learning
# greedy exploration
with open (os.path.join('./pkl','lander_27-09-2018_10-21-54.pkl'), 'rb') as fg:
    gre = pickle.load(fg)
gre_r, gre_t, gre_mean_r, gre_best_mean_r = gre['reward'], gre['timestep'], gre['mean_reward'], gre['best_reward']
# e-greedy exploration
with open (os.path.join('./pkl','lander_05-12-2018_15-05-26_soft.pkl'), 'rb') as fe:
    egr = pickle.load(fe)
egr_r, egr_t, egr_mean_r, egr_best_mean_r = egr['reward'], egr['timestep'], egr['mean_reward'], egr['best_reward']
# boltzmann exploration
with open (os.path.join('./pkl','lander_05-12-2018_16-28-12_0.001.pkl'), 'rb') as fbo:
    bol = pickle.load(fbo)
bol_r, bol_t, bol_mean_r, bol_best_mean_r = bol['reward'], bol['timestep'], bol['mean_reward'], bol['best_reward']
# bayesian exploration
with open (os.path.join('./pkl','lander_05-12-2018_17-34-38_0.0001.pkl'), 'rb') as fba:
    bay = pickle.load(fba)
bay_r, bay_t, bay_mean_r, bay_best_mean_r = bay['reward'], bay['timestep'], bay['mean_reward'], bay['best_reward']


# Plot result
sns.set(style="darkgrid", font_scale=1.5)
f, ax = plt.subplots(1, 1)
ax.plot(gre_t, gre_mean_r,      color="blue",  label="mean_vanilla_dqn",    linestyle="-")
ax.plot(gre_t, gre_best_mean_r, color="blue",  label="best_vanilla_dqn",    linestyle="-.")
ax.plot(egr_t, egr_mean_r,      color="red",   label="mean_soft_dqn",       linestyle="-")
ax.plot(egr_t, egr_best_mean_r, color="red",   label="best_soft_dqn",       linestyle="-.")
ax.plot(bol_t, bol_mean_r,      color="green", label="mean_soft_dqn_ex2_coef1e-3", linestyle="-")
ax.plot(bol_t, bol_best_mean_r, color="green", label="best_soft_dqn_ex2_coef1e-3", linestyle="-.")
ax.plot(bay_t, bay_mean_r,      color="black", label="mean_soft_dqn_ex2_coef1e-4",  linestyle="-")
ax.plot(bay_t, bay_best_mean_r, color="black", label="best_soft_dqn_ex2_coef1e-4",  linestyle="-.")
ax.legend()

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel("Episode Rewards", fontsize=15)
plt.xlabel("Number of Timesteps", fontsize=15, labelpad=4)
plt.title ("Soft Q Learning with EX2 Classifier Performance for Lander", fontsize=20)
plt.legend(loc='lower right')
plt.show()

