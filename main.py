import gym
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.model_selection import ParameterGrid

#Save logs to file
log_path = ('Logs')

#Create an environment
env = gym.make("CarRacing-v0")

#Variables for grid search
l_rate = [0.0001, 0.001, 0.01, 0.1]
policy = ["MlpPolicy", "CnnPolicy"]
n_steps = [1024, 2048, 4096]
n_epochs = [5, 10, 20]
highscore = -10000

param_grid = {'l_rate': [0.0001, 0.001, 0.01, 0.1], 'policy' : ["MlpPolicy", "CnnPolicy"], 'n_steps': [1024, 2048, 4096], 'n_epochs': [5, 10, 20], 'timesteps': [1000, 10000, 100000]}

grid = ParameterGrid(param_grid)

#Grid seatch on model
for params in grid:
	model = PPO(params['policy'], env, verbose=0, learning_rate=params['l_rate'],tensorboard_log=log_path, seed=2137, n_steps=params['n_steps'], n_epochs=params['n_epochs'])

	#Perform learning procedure
	print("LEARNING")
	model.learn(total_timesteps=params['timesteps'])

	#Evaluate learning outcomes
	print("EVALUATING")
	#Returns (Mean reward, Standard deviation)
	#Change render to true to see results
	mean, std = evaluate_policy(model, env, n_eval_episodes=3, render=False)
	print("Result: ",mean)
	#Check if it is currently the best model
	if mean > highscore:
		highscore = mean
		print("New highscore: ", highscore)
		#Save the best model to file in a .zip format
		print("SAVING TO FILE")
		model.save('Logs/Model')

''' 
#THIS IS HOW LOADING MODEL FROM FILE WOULD LOOK LIKE
model.load('Logs/Model')
'''

#Close environment
env.close()
