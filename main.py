import gym
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


BARTEK = 2137
PAWEL = 1488


env = gym.make("CarRacing-v0")
episodes = 5
'''
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0 

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode+1, score))
env.close()
'''
log_path = os.path.join('Training', 'Logs')
model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.001488,tensorboard_log=log_path, seed=PAWEL)
model.learn(total_timesteps=40000)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

'''
observation_space = env.observation_space
action_space = env.action_space
print(observation_space)
print(action_space)

env_screen = env.render(mode = 'rgb_array')
env.close()
plt.imshow(env_screen)
plt.show()
'''
