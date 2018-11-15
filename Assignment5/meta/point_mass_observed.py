import numpy as np
from gym import spaces
from gym import Env


class ObservedPointEnv(Env):
    """
    point mass on a 2-D plane
    four tasks: move to (-10, -10), (-10, 10), (10, -10), (10, 10)

    Problem 1: augment the observation with a one-hot vector encoding the task ID
     - change the dimension of the observation space
     - augment the observation with a one-hot vector that encodes the task ID
    """
    #====================================================================================#
    #                           ----------PROBLEM 1----------
    #====================================================================================#
    # YOUR CODE SOMEWHERE HERE
    def __init__(self, num_tasks=1):
        self.tasks = [0, 1, 2, 3][:num_tasks]
        self.task_idx = -1
        self.reset_task()
        self.reset()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2+num_tasks,))
        print('space is',self.observation_space)
        # exit()
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, is_evaluation=False):
        # for evaluation, cycle deterministically through all tasks
        if is_evaluation:
            self.task_idx = (self.task_idx + 1) % len(self.tasks)
        # during training, sample tasks randomly
        else:
            self.task_idx = np.random.randint(len(self.tasks))
        # self.task is [0,1,2,3], self.task_idx is index and self._task is the chosen task
        self._task = self.tasks[self.task_idx]
        if 0:
            print(self.tasks)
            print(self.task_idx)
            print(self._task)
            exit()
        # Basically set the goal here
        goals = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        self._goal = np.array(goals[self.task_idx])*10

    def reset(self):
        # state is the observation, where indicates 2d coordinate
        self._state = np.array([0, 0], dtype=np.float32) 
        return self._get_obs()

    def _get_obs(self):
        arr = np.zeros(len(self.tasks))
        arr[self.task_idx] = 1.0
        state = np.copy(self._state)
        return np.concatenate((state, arr), axis=0)

    def step(self, action):
        # Array can also be written as follows.
        # self._state represent 2d coordinate, x and y
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        # check if task is complete
        done = abs(x) < 0.01 and abs(y) < 0.01
        # move to next state
        # action is the 2d motion field
        self._state = self._state + action
        ob = self._get_obs()
        if 0:
            print('after',ob)
            print(np.shape(ob))
            exit()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed
