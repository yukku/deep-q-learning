import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import load_model
from time import time
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
import os.path
from environment import Environment
 
EPISODES = 3
WINDOW_SIZE = 10
DATA_SET_PATH = "datasets/NVDA Aug 24 2018.csv"
WEIGHTS_NAME = "23.08.18/dqn_weights.h5f"

env = Environment(dataSetPath=DATA_SET_PATH)
model = Sequential()
model.add(Flatten(input_shape=(WINDOW_SIZE,) + env.observation_space.shape))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(env.action_space.n))
model.add(Activation('linear'))

memory = SequentialMemory(
  limit=env.dataLength, 
  window_length=WINDOW_SIZE
)
policy = BoltzmannQPolicy()
dqn = DQNAgent(
  model=model, 
  nb_actions=env.action_space.n, 
  memory=memory, 
  nb_steps_warmup=100, 
  target_model_update=1e-2, 
  policy=policy
)
dqn.compile(
  Adam(lr=0.001), 
  metrics=['mae']
)

# if(os.path.exists(WEIGHTS_NAME)):
#   dqn.load_weights(WEIGHTS_NAME)
#   print("saved weight loaded")

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

dqn.fit(
  env, 
  nb_steps=EPISODES*env.dataLength, 
  log_interval=EPISODES*env.dataLength,
  callbacks=[tensorboard]
  # visualize=True
)
dqn.test(env, nb_episodes=1, visualize=False)
dqn.save_weights(WEIGHTS_NAME, overwrite=True)
