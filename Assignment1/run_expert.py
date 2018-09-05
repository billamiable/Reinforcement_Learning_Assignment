#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from simulate import simulate

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    # Make result reproduciable
    seed_expert = 0
    np.random.seed(seed_expert)   

    # Run expert
    model_path, meta_path, is_dagger, save_oa = False, False, False, True
    save_path = os.path.join('expert_data', args.envname + '-' + str(args.num_rollouts) + '-seed' + str(seed_expert) + '.pkl')
    mean, std = None, None
    observations, actions, rewards = simulate(args.envname, args.render, args.max_timesteps, args.num_rollouts, args.expert_policy_file, model_path, meta_path, save_path, mean, std, seed_expert, is_dagger, save_oa)
    
    if 0:
        # Load policy
        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(args.expert_policy_file)
        # print(policy_fn)
        print('loaded and built')

        with tf.Session():
            tf_util.initialize()
            import gym
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit
            returns = []
            observations = []
            actions = []
            for i in range(args.num_rollouts):
                print('iter', i)
                # Generate random integer for each run
                oper_seed = np.random.randint(0,args.num_rollouts*10)
                # print(oper_seed)
                env.seed(oper_seed)
                obs = env.reset()
                done = False
                totalr = 0. 
                steps = 0
                while not done:
                    # start with initial observation and get output action
                    action = policy_fn(obs[None,:])
                    observations.append(obs)
                    action = action[0]
                    actions.append(action)
                    # Each timestep, the agent chooses an action, and the environment returns an observation and a reward.
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

            expert_data = {'observations': np.array(observations),
                          'actions': np.array(actions)}

            with open(os.path.join('expert_data', args.envname + '-' + str(args.num_rollouts) + '-seed' + str(seed) + '.pkl'), 'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL) # write observation and action to file

if __name__ == '__main__':
    main()
