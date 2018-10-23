import tensorflow as tf
import numpy as np

import utils


class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1):
        self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._init_dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3
        print('state  dim is',self._state_dim)
        print('action dim is',self._action_dim)
        print('action low  space is',self._action_space_low)
        print('action high space is',self._action_space_high)
        print('cost fun is', self._cost_fn)
        
        # Build the graph first
        self._sess, self._state_ph, self._action_ph, self._next_state_ph,\
            self._next_state_pred, self._loss, self._optimizer, self._best_action = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        state_ph      = tf.placeholder(shape=[None, self._state_dim],  dtype=tf.float32)
        # CHECKED FLOAT
        action_ph     = tf.placeholder(shape=[None, self._action_dim], dtype=tf.float32)
        next_state_ph = tf.placeholder(shape=[None, self._state_dim],  dtype=tf.float32)
        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state

            implementation details (in order):
                (a) Normalize both the state and action by using the statistics of self._init_dataset and
                    the utils.normalize function
                (b) Concatenate the normalized state and action
                (c) Pass the concatenated, normalized state-action tensor through a neural network with
                    self._nn_layers number of layers using the function utils.build_mlp. The resulting output
                    is the normalized predicted difference between the next state and the current state
                (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
                    the predicted next state

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        state_norm  = utils.normalize(state,  self._init_dataset.state_mean,  self._init_dataset.state_std)
        action_norm = utils.normalize(action, self._init_dataset.action_mean, self._init_dataset.action_std)
        nn_input = tf.concat([state_norm, action_norm], axis=1)
        # CHECKED SCOPE NOT USED
        nn_output = utils.build_mlp(nn_input, self._state_dim, scope='p1', n_layers=self._nn_layers, reuse=reuse)
        state_diff = utils.unnormalize(nn_output, self._init_dataset.delta_state_mean, self._init_dataset.delta_state_std)
        next_state_pred = state_diff + state
        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent

            implementation details (in order):
                (a) Compute both the actual state difference and the predicted state difference
                (b) Normalize both of these state differences by using the statistics of self._init_dataset and
                    the utils.normalize function
                (c) The loss function is the mean-squared-error between the normalized state difference and
                    normalized predicted state difference
                (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        diff_actual = next_state_ph   - state_ph
        diff_pred   = next_state_pred - state_ph
        norm_diff_actual = utils.normalize(diff_actual, self._init_dataset.delta_state_mean, 
                                                        self._init_dataset.delta_state_std)
        norm_diff_pred   = utils.normalize(diff_pred  , self._init_dataset.delta_state_mean, 
                                                        self._init_dataset.delta_state_std)
        loss = tf.nn.l2_loss(norm_diff_actual - norm_diff_pred)
        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)
        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])

            implementation details (in order):
                (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
                (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
                    each of length self._horizon
                (c) Starting from the input state, unroll each action sequence using your neural network
                    dynamics model
                (d) While unrolling the action sequences, keep track of the cost of each action sequence
                    using self._cost_fn
                (e) Find the action sequence with the lowest cost, and return the first action in that sequence

            Hints:
                # TO-DO: So cost function is built-in function?
                (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
                    2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
                    state or action size
                (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
                (iii) Use tf.random_uniform(...) to generate the random action sequences

        """
        ### PROBLEM 2
        ### YOUR CODE HERE
        # raise NotImplementedError
        costs = np.zeros(self._num_random_action_selection)
        # Alternative way
        # self.state_ph_batch = tf.ones([self._num_random_action_selection, 1]) * state_ph[0]
        if 0:
            print('initial cost',costs)
            print(np.shape(costs))
            exit()
        # iterate horizon times, i.e. 15
        for i in range(self._horizon):
            # Each time needs resampling
            action_ph = tf.random_uniform(shape=[self._num_random_action_selection, self._action_dim],
                                          minval=self._action_space_low,
                                          maxval=self._action_space_high,
                                          dtype=tf.float32)
            if i == 0:
                first_action_ph = action_ph
            # Checked dynamics function works
            next_state_pred_ph = self._dynamics_func(state_ph, action_ph, reuse=True)
            # DIM should be [num_random_action_selection]
            # TO-DO: why need next state
            cost = self._cost_fn(state_ph, action_ph, next_state_pred_ph)     
            costs += cost
            state_ph = next_state_pred_ph
            if 0:
                print(action_ph.get_shape())
                print(next_state_pred_ph.get_shape())
                print(cost.get_shape()) 
                print('costs are',costs)
                exit()  
        
        # dim should be [num_random_action_selection], i.e. 4096
        best_seq_id = tf.argmin(costs, axis=0)
        best_action = first_action_ph[best_seq_id]
        if 0:
            print(costs.get_shape())
            print(best_action.get_shape())
            exit()
        return best_action

    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        sess = tf.Session()

        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        state_ph, action_ph, next_state_ph = self._setup_placeholders()
        # reuse used when the trained model need to be fixed later on.
        next_state_pred = self._dynamics_func(state_ph, action_ph, reuse=False)
        loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)

        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action = self._setup_action_selection(state_ph)

        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, \
                next_state_pred, loss, optimizer, best_action
    
    # Definition here is used in another py file
    def train_step(self, states, actions, next_states):
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        # VIP!!! it is next_state_pred not next_state_ph
        _, loss = self._sess.run([self._optimizer, self._loss], feed_dict={self._state_ph: states, 
                                                                           self._action_ph: actions, 
                                                                           self._next_state_ph: next_states})
        return loss

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)

        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        next_state_pred = self._sess.run(self._next_state_pred, feed_dict={self._state_ph: [state], 
                                                                           self._action_ph: [action]})
        next_state_pred = next_state_pred[0]
        # print(np.shape(next_state_pred))
        # exit()
        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        # raise NotImplementedError
        # TO-DO: potentially speed up part       
        states = np.tile(state, (self._num_random_action_selection,1))
        best_action = self._sess.run(self._best_action, feed_dict={self._state_ph: states})

        # Option 2
        # best_action = self._sess.run(self._best_action, feed_dict={self._state_ph: [state]})                                                        
        if 0:
            best_action, costs, seq_id, first_act, act = self._sess.run([self._best_action, 
                                                                    self.costs, 
                                                                    self.best_seq_id, 
                                                                    self.first_action_ph,
                                                                    self.action_ph], 
                                                                    feed_dict={self._state_ph: [state]})
            print(costs)
            print(seq_id)
            print('best costs',costs[seq_id])
            print('last act',act)
            print('first act',first_act)
            # print()
            print('best action', best_action)
            exit()
        assert np.shape(best_action) == (self._action_dim,)
        return best_action
