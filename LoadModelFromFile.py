import gym
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack

#Create an environment
env = gym.make("CarRacing-v0")

#Prepare model
model = PPO('CnnPolicy', env, verbose=1, seed=2137)
	
model = model.load('Logs/Model350k')

while True:
	obs = env.reset()
	for _ in range(1000):
		action, _states = model.predict(obs.copy())
		obs, rewards, done, info = env.step(action)
		env.render()
