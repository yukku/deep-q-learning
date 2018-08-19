import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from environment import Environment
 
EPISODES = 20000

env = Environment()
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(env.action_space.n))
model.add(Activation('linear'))

memory = SequentialMemory(
  limit=env.dataLength, 
  window_length=1
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
  Adam(lr=1e-3), 
  metrics=['mae']
)

dqn.fit(
  env, 
  nb_steps=EPISODES*env.dataLength, 
  log_interval=EPISODES*env.dataLength,
  # visualize=True
)
# dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
# dqn.test(env, nb_episodes=5, visualize=False)
