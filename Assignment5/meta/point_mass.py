import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point mass on a 2-D plane
    goals are sampled randomly from a square
    """

    def __init__(self, num_tasks=1):
        self.reset_task()
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, is_evaluation=False, train_test=False, granularity=1):
        '''
        sample a new task randomly

        Problem 3: make training and evaluation goals disjoint sets
        if `is_evaluation` is true, sample from the evaluation set,
        otherwise sample from the training set
        '''
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        #====================================================================================#
        # YOUR CODE HERE
        # FOR DIFFERENT SETTING, TRAINNING IS ONE TASK, EVALUATION IS ANOTHER
        # TO-DO: How to construct chessboarder?
        # Actually do not need to construct, but mannually set 
        # print(train_test)
        # print(granularity)
        # exit()
        if train_test:
<<<<<<< HEAD
            print('running problem 3!')
=======
            # print('running problem 3!')
>>>>>>> a2caffb7e8a6af10b0acb9fd9e82c1da4502032d
            # Define x and y
            # Test
            if is_evaluation:
                if granularity==1:
                    x = np.random.uniform(-10, 10)
                    if x<=0:
                        y = np.random.uniform(-1, 0)
                        x = x + 2 * np.random.randint(-4,6)
                        y = y + 2 * np.random.randint(-4,6)
                    else:
                        y = np.random.uniform(0, 1)
                        x = x + 2 * np.random.randint(-5,5)
                        y = y + 2 * np.random.randint(-5,5)
                elif granularity==10:
                    x = np.random.uniform(-10, 10)
                    if x<=0:
                        y = np.random.uniform(-10, 0)
                    else:
                        y = np.random.uniform(0, 10)
            # Train
            else:
                if granularity==1:
<<<<<<< HEAD
                    print('using granularity 1')
=======
                    # print('using granularity 1')
>>>>>>> a2caffb7e8a6af10b0acb9fd9e82c1da4502032d
                    x = np.random.uniform(-10, 10)
                    if x<=0:
                        y = np.random.uniform(0, 1)
                        x = x + 2 * np.random.randint(-4,6)
                        y = y + 2 * np.random.randint(-5,5)
                    else:
                        y = np.random.uniform(-1, 0)
                        x = x + 2 * np.random.randint(-5,5)
                        y = y + 2 * np.random.randint(-4,6)
                elif granularity==10:
<<<<<<< HEAD
                    print('using granularity 10')
=======
                    # print('using granularity 10')
>>>>>>> a2caffb7e8a6af10b0acb9fd9e82c1da4502032d
                    x = np.random.uniform(-10, 10)
                    if x<=0:
                        y = np.random.uniform(0, 10)
                    else:
                        y = np.random.uniform(-10, 0)
            self._goal = np.array([x, y])
            # print('actually get in')
            # print(self._goal)
            # exit()
        else:
<<<<<<< HEAD
            print('not running problem 3!')
=======
            # print('not running problem 3!')
>>>>>>> a2caffb7e8a6af10b0acb9fd9e82c1da4502032d
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            self._goal = np.array([x, y])
            # print(self._goal)
            # exit()

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2) ** 0.5

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # check if task is complete
        done = abs(x) < .01 and abs(y) < .01
        reward = self.reward_function(x, y)
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed
