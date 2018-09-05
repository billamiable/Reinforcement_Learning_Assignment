import numpy as np
import tensorflow as tf
import pickle

def preprocess(observations, actions, val_split, normalization, shuffle):
    # Normalization
    if normalization:
      mean = np.mean(observations, axis=0)
      std = np.std(observations, axis=0) + 1e-6
      observations = (observations - mean) / std
      if 0:
          a = np.array([[2,3],[2,2],[1,4]])
          print(np.mean(a, axis=0))
          print(np.std(a, axis=0))
          b = (a -np.mean(a, axis=0))/ np.std(a, axis=0)
          print(b)
          c = b * np.std(a, axis=0) + np.mean(a, axis=0)
          print(c)
          exit()

    # Shuffle the dataset
    if shuffle:
        indices = np.arange(observations.shape[0])
        np.random.shuffle(indices)
        observations = observations[indices]
        actions = actions[indices]
        if 0:
            observations = observations[0:4]
            actions = actions[0:4]
            print(observations.shape)
            print(actions.shape)

    # Split dataset
    total_sample = len(observations)
    print('total sample size is ',total_sample)
    obs_train = observations[0:round(total_sample * val_split)]
    act_train = actions[0:round(total_sample * val_split)]  
    obs_test = observations[round(total_sample * val_split):]
    act_test = actions[round(total_sample * val_split):]
    train_num = len(obs_train)
    test_num = len(obs_test)
    print('training set size is ',train_num)
    print('test set size is ',test_num)

    return obs_train, act_train, obs_test, act_test, train_num, test_num, mean, std

def next_batch(observation, action, batch_size):
    indices = np.random.randint(low=0, high=len(observation), size=batch_size)
    # print(indices)
    batch_x = observation[indices]
    batch_y = action[indices]
    return batch_x, batch_y

def training(data, label, restore, cost, pred, optimizer, val_split, normalization, shuffle, epochs, batch_size, display_step, model_save_path, loss_save_path):
    # if not restore: ？？
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    # saver = tf.train.import_meta_graph(restore)
    # Get data
    obs_train, act_train, obs_test, act_test, train_num, test_num, mean, std = preprocess(data, label, val_split, normalization, shuffle)
    with tf.Session() as sess:
        if restore:
            # Restore variables from disk.
            saver.restore(sess, restore)
            print("Model restored.")
        else:
            sess.run(init)
            print("Model initialized.")

        # Feed data
        x = sess.graph.get_tensor_by_name('x:0')  # Get tensor
        y = sess.graph.get_tensor_by_name('y:0')  # Get tensor
        history, history_test = [], []
        print('start training...')
        for epoch in range(epochs):
            avg_cost, avg_cost_test = 0., 0. # Initialization
            total_batch = int(train_num/batch_size)
            test_batch = int(test_num/batch_size)
            # Train
            for i in range(total_batch):
                batch_xs, batch_ys = next_batch(obs_train, act_train, batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += c / total_batch # Compute average loss
            history.append(avg_cost)

            # Test
            for j in range(test_batch):
                batch_xs_test, batch_ys_test = next_batch(obs_test, act_test, batch_size)
                c_test =  sess.run(cost, feed_dict={x: batch_xs_test, y: batch_ys_test})
                avg_cost_test += c_test / test_batch
            history_test.append(avg_cost_test)
            
            # Display loss
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "Training cost=", "{:.9f}".format(avg_cost))
                if (epoch+1) % (5*display_step) == 0:
                    print("Epoch:", '%04d' % (epoch+1), "Test cost=", "{:.9f}".format(avg_cost_test))

        # Store trained model
        save_path = saver.save(sess, model_save_path)
        print("Model saved in path: %s" % save_path)

        # Save loss data into pkl file
        loss_data = {'train': np.array(history), 'test': np.array(history_test)}
        with open(loss_save_path, 'wb') as fp:
            pickle.dump(loss_data, fp)

    return loss_data, mean, std