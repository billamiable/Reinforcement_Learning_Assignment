import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def simulate(envname, render, max_timesteps, num_rollouts, policy_path, model_path, meta_path, save_path, mean, std, seed, is_dagger, save_oa):
    # Make result reproducible
    np.random.seed(seed)
    
    # Load policy
    if policy_path:
        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(policy_path)
        # print(policy_fn)
        print('loaded and built')
    
    # Restore model graph
    if model_path:
        saver = tf.train.import_meta_graph(meta_path)
        pred = tf.get_collection("training_collection")[1]

    with tf.Session() as sess:
        # Only initialize variables
        tf_util.initialize()
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit
        rewards = []
        observations = []
        actions = []
        # DAgger needs to save expert actions for future use
        if is_dagger:
            actions_expert = []

        if model_path:
            # Restore variables from disk.
            saver.restore(sess, model_path)
            x = sess.graph.get_tensor_by_name('x:0')
        
        for i in range(num_rollouts):
            # Generate operational seed
            oper_seed = np.random.randint(0,num_rollouts*10)
            if not is_dagger:
                print('iter', i)
            # print(oper_seed)
            env.seed(oper_seed)
            obs = env.reset()
            done = False
            totalr = 0. 
            steps = 0
            while not done:
                obs = obs[None,:]
                # print(np.shape(obs))
                if policy_path:
                      if is_dagger:
                          action_expert = policy_fn(obs)
                      else:
                          action = policy_fn(obs)
                # need to append unnormalized obs
                observations.append(obs[0])
                # print(np.shape(obs[0]))
                if model_path:
                    obs = (obs - mean) / std
                    action = sess.run(pred, feed_dict={x: obs})
                action = action[0]
                actions.append(action)
                # print(np.shape(action))
                if is_dagger:
                    actions_expert.append(action_expert[0])
                # Each timestep, the agent chooses an action, and the environment returns an observation and a reward.
                obs, r, done, _ = env.step(action) # ??input shape
                totalr += r
                steps += 1
                if render:
                    env.render()
                if not is_dagger:
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            rewards.append(totalr)

        print('rewards', rewards)
        print('mean reward', np.mean(rewards))
        print('std of reward', np.std(rewards))

        # File limit concern
        if save_oa:
            expert_data = {'rewards': np.array(rewards), 'observations': np.array(observations), 'actions': np.array(actions)}
        else:
            expert_data = {'rewards': np.array(rewards)}

        # Save to file
        with open(save_path, 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL) 
    
    if is_dagger:
        return observations, actions, rewards, actions_expert
    else:
        return observations, actions, rewards