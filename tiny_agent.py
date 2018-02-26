from collections import deque
import gym
import numpy as np
import os
import tensorflow as tf

done = True  # env needs to be reset

# First let's build the two DQNs (online & target)
input_height = 88 #84
input_width = 80 #84
input_channels = 1 #4
conv_n_maps = [32, 64, 64] 
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available
initializer = tf.contrib.layers.variance_scaling_initializer()


class tinyAgent(object):

	def __init__(self, num_actions, input_size, hidden_layer_size, learning_rate,gamma,decay_rate,greedy_e_epsilon,random_seed):
		# store hyper-params
		
		self.input_height = 88
		self.self_input_width = 80
		self.input_channels = 1
		self.conv_n_maps = [3, 64, 64]
		self.kernel_sizes = [(8,8), (4,4), (3,3)]
		self.conv_paddings = [4,2,1]
		self.conv_activation = ["SAME"] * 3
		self.n_hidden_in = [tf.nn.relu] * 3 
		self.n_hidden_in = 64 * 11 * 11
		self.hidden_activation = tf.nn.relu
		self.n_outputs = num_actions
		self.initializer = tf.contrib.layers.variance_scaling_initializer()
		self.learning_rate = learning_rate
		self.momentum = 0.95
		self.replay_memory_size
		self.training_start
		self.discount_rate
		self.skip_start = 90
		self.batch_size = 50
		self.iteration = 0
		self.done = True
		self.steps = 4000000
		self.color =  np.array([210, 164, 74]).mean()


		self.eps_min = 0.1
		self.eps_max = 1.0
		self.eps_decay_steps = self.steps // 2


		#TODO EPSILO DECAY WITH STEPS


		self._A = num_actions
		self._D = input_size
		self._H = hidden_layer_size
		#self._learning_rate = learning_rate
		self._decay_rate = decay_rate
		self._gamma = gamma
		
		# some temp variables
		self._xs,self._hs,self._dlogps,self._drs = [],[],[],[]

		# variables governing exploration
		self._exploration = True # should be set to false when evaluating
		self._explore_eps = greedy_e_epsilon
		

		online_q_values, online_vars = q_network(X_state, name="q_networks/online")
		target_q_values, target_vars = q_network(X_state, name="q_networks/target")

		with tf.variable_scope("train"):
		    X_action = tf.placeholder(tf.int32, shape=[None])
		    y = tf.placeholder(tf.float32, shape=[None, 1])
		    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
		                            axis=1, keep_dims=True)
		    error = tf.abs(y - q_value)
		    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
		    linear_error = 2 * (error - clipped_error)
		    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

		    global_step = tf.Variable(0, trainable=False, name='global_step')
		    optimizer = tf.train.MomentumOptimizer(
		        learning_rate, momentum, use_nesterov=True)
		    training_op = optimizer.minimize(loss, global_step=global_step)

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()








	def q_network(X_state, name):
	    prev_layer = X_state
	    with tf.variable_scope(name) as scope:
	        for n_maps, kernel_size, strides, padding, activation in zip(
	                conv_n_maps, conv_kernel_sizes, conv_strides,
	                conv_paddings, conv_activation):
	            prev_layer = tf.layers.conv2d(
	                prev_layer, filters=n_maps, kernel_size=kernel_size,
	                strides=strides, padding=padding, activation=activation,
	                kernel_initializer=initializer)
	        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
	        hidden = tf.layers.dense(last_conv_layer_flat, self._H,
	                                 activation=hidden_activation,
	                                 kernel_initializer=initializer)
	        outputs = tf.layers.dense(hidden, n_outputs,
	                                  kernel_initializer=initializer)
	    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
	                                       scope=scope.name)
	    trainable_vars_by_name = {var.name[len(scope.name):]: var
	                              for var in trainable_vars}
	    return outputs, trainable_vars_by_name

	X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
	                                            input_channels])


	online_q_values, online_vars = q_network(X_state, name="q_networks/online")
	target_q_values, target_vars = q_network(X_state, name="q_networks/target")

	# We need an operation to copy the online DQN to the target DQN
	copy_ops = [target_var.assign(online_vars[var_name])
	            for var_name, target_var in target_vars.items()]
	copy_online_to_target = tf.group(*copy_ops)

	# Now for the training operations
	learning_rate = 0.001
	momentum = 0.95

	# initialize q with random weights
	
	replay_memory_size = 20000 # 1000000
	replay_memory = deque([], maxlen=replay_memory_size)

	def sample_memories(batch_size):
	    indices = np.random.permutation(len(replay_memory))[:batch_size]
	    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
	    for idx in indices:
	        memory = replay_memory[idx]
	        for col, value in zip(cols, memory):
	            col.append(value)
	    cols = [np.array(col) for col in cols]
	    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
	           cols[4].reshape(-1, 1))

	# And on to the epsilon-greedy policy with decaying epsilon
	eps_min = 0.1
	eps_max = 1.0 if not args.test else eps_min
	eps_decay_steps = args.number_steps // 2

	def epsilon_greedy(q_values, step):
	    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
	    if np.random.rand() < epsilon:
	        return np.random.randint(n_outputs) # random action
	    else:
	        return np.argmax(q_values) # optimal action


	# We need to preprocess the images to speed up training
	mspacman_color = np.array([210, 164, 74]).mean()

	def preprocess_observation(q):
	    first = q.popleft()
	    second = q.popleft()
	    third = q.popleft()
	    fourth = q.popleft()
	    img = first[1:176:2, ::2] # crop and downsize
	    img2 = second[1:176:2, ::2]
	    # print (img)
	    # print ("AAAAAAA\n\n")
	    # print (img2)
	    # print ("BBBBBBB\n\n")
	    # print (img2 - img)
	    
	    #img = img - img2
	    img3 = third[1:176:2, ::2]
	    img4 = fourth[1:176:2, ::2]
	    img = img2 + img
	    img3 = img4 + img3
	    img = img + img3
	    img = img.mean(axis=2) # to greyscale
	    img[img==mspacman_color] = 0 # Improve contrast
	    #img = (img - 128) / 128 - 1 # normalize from -1. to 1.
	    #print (img)
	    #sys.exit()
	    #img4 = img4.mean(axis=2) # to greyscale
	    #img4[img4==mspacman_color] = 0 # Improve contrast
	    #img4 = (img4 - 128) / 128 - 1

	    #img = np.array([88,80])
	    #img3 = np.array([88,80])

	    #img = img - img3
	    img = (img - 128) / 128 - 1
	    #res = np.array(img,img3)
	    q.append(first)
	    q.append(second)
	    q.append(third)
	    q.append(fourth)

	    return img.reshape(88, 80, 1)