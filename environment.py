import sys
import gym
import gym.spaces
import numpy as np
from utils import getDataset, getState, formatPrice

class Environment(gym.core.Env):
  def __init__(self, dataSetPath):
    self.action_space = gym.spaces.Discrete(3)
    self.observation_space = gym.spaces.Box(
      low=np.array([0.0]), 
      high=np.array([1.0])
    )
    self.action = ["hold", "buy", "sell"]
    self.counter = 0
    self.data = getDataset(dataSetPath)
    self.dataLength = len(self.data)
    self.inventory = []
    self.total_profit = 0

  def getActionNameByIndex(self, index):
    return self.action[index]

  def step(self, action):

    self.counter += 1
    state = getState(self.data, self.counter, 1 + 1)
    # print(state)
    reward = 0

    if action == 0: #hold
      print(str(self.counter) + " - Hold")

    if action == 1: # buy
      self.inventory.append(self.data[self.counter])
      print(str(self.counter) + " - Buy: " + formatPrice(self.data[self.counter]))

    elif action == 2 and len(self.inventory) > 0: # sell
      bought_price = self.inventory.pop(0)
      reward = max(self.data[self.counter] - bought_price, 0)
      self.total_profit += self.data[self.counter] - bought_price
      print(str(self.counter) + " - Sell: " + formatPrice(self.data[self.counter]) + " | Profit: " + formatPrice(self.data[self.counter] - bought_price))

    done = (self.counter == self.dataLength - 1)
    if(done): 
      print("\nTOTAL_PROFIT: " + formatPrice(self.total_profit) + "\n")

    return np.array(state), reward, done, {}
 
  def reset(self):
    self.counter = 0
    self.inventory = []
    self.total_profit = 0
    state = getState(self.data, 0, 1 + 1)
    return np.array(state)

  def getLatestState(self, index, window):
    return getState(self.data, self.dataLength - 1 + index, window + 1)


