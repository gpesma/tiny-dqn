from __future__ import division, print_function, unicode_literals
from collections import deque
import gym
import numpy as np
import os
import tensorflow as tf

done = True  # env needs to be reset

# First let's build the two DQNs (online & target)
# input_height = 88 #84
# input_width = 80 #84
# input_channels = 1 #4
# conv_n_maps = [32, 64, 64] 
# conv_kernel_sizes = [(8,8), (4,4), (3,3)]
# conv_strides = [4, 2, 1]
# conv_paddings = ["SAME"] * 3 
# conv_activation = [tf.nn.relu] * 3
# n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
# n_hidden = 512
# hidden_activation = tf.nn.relu
# n_outputs = env.action_space.n  # 9 discrete actions are available
# initializer = tf.contrib.layers.variance_scaling_initializer()


class tinyAgent(object):




	def __init__(self, num_actions,learning_rate):
			
		self.input_height = 88
		self.input_width = 80
		self.input_channels = 1
		self.conv_n_maps = [3, 64, 64]
		self.conv_kernel_sizes = [(8,8), (4,4), (3,3)]
		self.conv_strides = [4,2,1]
		self.conv_paddings = ["SAME"] * 3
		self.conv_activation = [tf.nn.relu] * 3
		self.n_hidden = 512 
		self.n_hidden_in = 64 * 11 * 10
		self.hidden_activation = tf.nn.relu
		self.n_outputs = num_actions
		self.initializer = tf.contrib.layers.variance_scaling_initializer()
		self.learning_rate = learning_rate
		self.momentum = 0.95
		self.replay_memory_size = 20000
		self.training_start = 10000
		self.discount_rate = 0.99
		self.skip_start = 90
		self.batch_size = 50
		self.iteration = 0
		self.done = True
		self.steps = 4000000
		self.color =  np.array([210, 164, 74]).mean()
		self.replay_memory = deque([], maxlen=self.replay_memory_size)

		self.eps_min = 0.1
		self.eps_max = 1.0
		self.eps_decay_steps = self.steps // 2

		self.X_state = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width,
	                                            self.input_channels])



		def q_network(X_state, name):
		    prev_layer = X_state
		    with tf.variable_scope(name) as scope:
		        for n_maps, kernel_size, strides, padding, activation in zip(
		                self.conv_n_maps, self.conv_kernel_sizes, self.conv_strides,
		                self.conv_paddings, self.conv_activation):
		            prev_layer = tf.layers.conv2d(
		                prev_layer, filters=n_maps, kernel_size=kernel_size,
		                strides=strides, padding=padding, activation=activation,
		                kernel_initializer=self.initializer)
		        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, self.n_hidden_in])
		        hidden = tf.layers.dense(last_conv_layer_flat, self.n_hidden,
		                                 activation=self.hidden_activation,
		                                 kernel_initializer=self.initializer)
		        outputs = tf.layers.dense(hidden, self.n_outputs,
		                                  kernel_initializer=self.initializer)
		    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
		                                       scope=scope.name)
		    trainable_vars_by_name = {var.name[len(scope.name):]: var
		                              for var in trainable_vars}
		    return outputs, trainable_vars_by_name




		self.online_q_values, online_vars = q_network(self.X_state, name="q_networks/online")
		self.target_q_values, target_vars = q_network(self.X_state, name="q_networks/target")


		# We need an operation to copy the online DQN to the target DQN
		copy_ops = [target_var.assign(online_vars[var_name])
	            for var_name, target_var in target_vars.items()]
		copy_online_to_target = tf.group(*copy_ops)

		with tf.variable_scope("train"):
		    X_action = tf.placeholder(tf.int32, shape=[None])
		    y = tf.placeholder(tf.float32, shape=[None, 1])
		    q_value = tf.reduce_sum(self.online_q_values * tf.one_hot(X_action, self.n_outputs),
		                            axis=1, keep_dims=True)
		    error = tf.abs(y - q_value)
		    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
		    linear_error = 2 * (error - clipped_error)
		    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

		    global_step = tf.Variable(0, trainable=False, name='global_step')
		    optimizer = tf.train.MomentumOptimizer(
		        self.learning_rate, self.momentum, use_nesterov=True)
		    training_op = optimizer.minimize(loss, global_step=global_step)

		self.gl_step = global_step
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()

		copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
		self.copy_online_to_target = tf.group(*copy_ops)









	def sample_memories(batch_size):
	    indices = np.random.permutation(len(self.replay_memory))[:batch_size]
	    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
	    for idx in indices:
	        memory = self.replay_memory[idx]
	        for col, value in zip(cols, memory):
	            col.append(value)
	    cols = [np.array(col) for col in cols]
	    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
	           cols[4].reshape(-1, 1))


	def epsilon_greedy(q_values, step):
	    self.epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * self.step/self.eps_decay_steps)
	    if np.random.rand() < epsilon:
	        return np.random.randint(n_outputs) # random action
	    else:
	        return np.argmax(q_values) # optimal actionage


	def learn(self, sess):
	    X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))
        next_q_values = self.target_q_values.eval(
            feed_dict={self.X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train the online DQN
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val, X_action: X_action_val, y: y_val})


        # Regularly copy the online DQN to the target DQN
        if step % args.copy_steps == 0:
            copy_online_to_target.run()

        # And save regularly
        if step % args.save_steps == 0:
            saver.save(sess, args.path)	



	# We need to preprocess the images to speed up training
	

