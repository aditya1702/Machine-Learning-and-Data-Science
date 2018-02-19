import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class RandomWalkEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.action_space = spaces.Discrete(2)
    self.size = 6
    #print("init")
  def _step(self, action):
    #print("step")
    reward = 0
    done = False
    if (action == 0):
       self.state -= 1
    if (action == 1):
        self.state += 1
    if (self.state >= self.size):
        reward = 1
        done = True
    if (self.state <= 0):
        done = True
    return np.array(self.state), reward, done, {}
  def _reset(self):
    #print("reset")
    print("#self.size:",self.size)
    self.state =  np.random.randint(1,self.size-1)
    print("starting: ", self.state)
  def _render(self, mode='human', close=False):
    if close:
        return
    #print("render")
    print("current state: ",self.state)