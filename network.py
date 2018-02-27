import gym
import numpy as np
import os
import tensorflow as tf
import csv
import sys
from tiny_agent import tinyAgent

env = gym.make("KungFuMaster-v0")
num_a = 0
for i in range(1,20) :
	if env.action_space.contains(i) :
		num_a += 1
	else :
		break

def preprocess_observation(obs, prev_obs):
    mspacman_color = np.array([210, 164, 74]).mean()
    img = obs[1:176:2, ::2] # crop and downsize
    img2 = prev_obs[1:176:2, ::2]
    img = img + img2
    img = img.mean(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
 
    img = (img - 128) / 128 - 1
    return img.reshape(88, 80, 1)

def epsilon_greedy(q_values, step, agent):
    agent.epsilon = max(agent.eps_min, agent.eps_max - (agent.eps_max-agent.eps_min) *  step/agent.eps_decay_steps)
    if np.random.rand() < agent.epsilon:
        return np.random.randint(agent.n_outputs) # random action
    else:
        return np.argmax(q_values)

learning_rate = 0.001
skip_start = 90
path = ""
agent = tinyAgent(num_a, learning_rate)
prev_obs = None
iteration = 0
games_played = 0
done = True
training_start = 10000
learn_iterations = 6
with tf.Session() as sess:
    if os.path.isfile(path + ".index"):
        saver.restore(sess, path)
    else:
        agent.init.run()
        agent.copy_online_to_target.run()


    while True:


    	step = agent.gl_step.eval()
    	iteration += 1
    	if iteration == 1:
        	prev_obs = env.reset()

        if done: # game over, start again
            obs = env.reset()
            for skip in range(skip_start): # skip the start of each game
                obs, reward, done, info = env.step(0)
            games_played = games_played + 1
            prev_obs = obs
            state = preprocess_observation(obs, prev_obs)


        env.render()

        q_values = agent.online_q_values.eval(feed_dict={agent.X_state: [state]})
        action = epsilon_greedy(q_values, step, agent)
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs,prev_obs)
        prev_obs = obs

        agent.replay_memory.append((state, action, reward, next_state, 1.0 - agent.done))
        state = next_state

        if iteration < training_start or iteration % learn_iterations != 0:
            continue