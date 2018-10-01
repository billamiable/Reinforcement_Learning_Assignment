import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# All using vanilla Q learning
# greedy exploration
with open (os.path.join('./pkl','atari_30-09-2018_14-55-51_greedy.pkl'), 'rb') as fg:
    gre = pickle.load(fg)
gre_r, gre_t, gre_mean_r, gre_best_mean_r = gre['reward'], gre['timestep'], gre['mean_reward'], gre['best_reward']
# e-greedy exploration
with open (os.path.join('./pkl','atari_28-09-2018_04-23-28_vanilla.pkl'), 'rb') as fe:
    egr = pickle.load(fe)
egr_r, egr_t, egr_mean_r, egr_best_mean_r = egr['reward'], egr['timestep'], egr['mean_reward'], egr['best_reward']
# boltzmann exploration
with open (os.path.join('./pkl','atari_30-09-2018_09-20-01_boltzmann.pkl'), 'rb') as fbo:
    bol = pickle.load(fbo)
bol_r, bol_t, bol_mean_r, bol_best_mean_r = bol['reward'], bol['timestep'], bol['mean_reward'], bol['best_reward']
# bayesian exploration
with open (os.path.join('./pkl','atari_01-10-2018_02-09-06_bayesian.pkl'), 'rb') as fba:
    bay = pickle.load(fba)
bay_r, bay_t, bay_mean_r, bay_best_mean_r = bay['reward'], bay['timestep'], bay['mean_reward'], bay['best_reward']


# Plot result
sns.set(style="darkgrid", font_scale=1.5)
f, ax = plt.subplots(1, 1)
ax.plot(gre_t, gre_mean_r,      color="blue",  label="mean_greedy",    linestyle="-")
ax.plot(gre_t, gre_best_mean_r, color="blue",  label="best_greedy",    linestyle="-.")
ax.plot(egr_t, egr_mean_r,      color="red",   label="mean_e-greedy",  linestyle="-")
ax.plot(egr_t, egr_best_mean_r, color="red",   label="best_e-greedy",  linestyle="-.")
ax.plot(bol_t, bol_mean_r,      color="green", label="mean_boltzmann", linestyle="-")
ax.plot(bol_t, bol_best_mean_r, color="green", label="best_boltzmann", linestyle="-.")
ax.plot(bay_t, bay_mean_r,      color="black", label="mean_bayesian",  linestyle="-")
ax.plot(bay_t, bay_best_mean_r, color="black", label="best_bayesian",  linestyle="-.")
ax.legend()

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel("Episode Rewards", fontsize=15)
plt.xlabel("Number of Timesteps", fontsize=15, labelpad=4)
plt.title ("Vanilla Q Learning Performance with Different Exploration Strategy for Pong", fontsize=20)
plt.legend(loc='lower right')
plt.show()

