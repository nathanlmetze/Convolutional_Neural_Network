import numpy as np
import gym
import random
import tensorflow as tf
import cv2
import datetime
import sys
import getopt
import matplotlib.pyplot as plt

GAME = "Enduro-v0"
INSTANCES = 5000
MAX_MEM = 200000
TIMER = 50000
SIZE_QLEARNER = 4
MINI_BATCH_SIZE = 32

class custom_agent:
    def __init__(self, game_env):
        self._actions = game_env.action_space.n
        self._memory = []
        self._gamma = 0.95 # NOTE Discount rate
        self._epsilon = 1.0 # NOTE Exploration rate
        self._epsilon_min = 0.01
        self._epsilon_decay = 0.99
        self._frames_remembered = 0
        self._load = False

        self._model, self._input_layer = self._construct_model(0)
        self._target_model, self._target_input_layer = self._construct_model(1)
        self._train()

        '''
        Code taken from TensorFlow Documentation
        https://www.tensorflow.org/api_docs/python/tf/InteractiveSession
        '''
        self._session = tf.InteractiveSession()
        self._session.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()
        self._session_network = [self._W_conv1T.assign(self._W_conv1),
            self._b_conv1T.assign(self._b_conv1),
            self._W_conv2T.assign(self._W_conv2),
            self._b_conv2T.assign(self._b_conv2),
            self._W_conv3T.assign(self._W_conv3),
            self._b_conv3T.assign(self._b_conv3),
            self._W_fc1T.assign(self._W_fc1),
            self._b_fc1T.assign(self._b_fc1),
            self._W_fc2T.assign(self._W_fc2),
            self._b_fc2T.assign(self._b_fc2)]



    '''
    Code taken from TensorFlow Documentation
    https://www.tensorflow.org/get_started/mnist/pro
    https://www.tensorflow.org/api_docs/python/tf/matmul
    https://www.tensorflow.org/api_docs/python/nn/relu
    https://www.tensorflow.org/tutorials/deep_cnn
    '''
    def _construct_model(self, which_model):
        #Tensorflow implementation
        input_layer = tf.placeholder("float", [None, 84, 95, 4], name =  "input_layer")
        # network weights
        '''
        Network created using guidelines from:
        http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html?foxtrotcallback=true
        '''
        W_conv1 = self._weight_variable([8,8,4,32])
        b_conv1 = self._bias_variable([32])

        W_conv2 = self._weight_variable([4,4,32,64])
        b_conv2 = self._bias_variable([64])

        W_conv3 = self._weight_variable([3,3,64,64])
        b_conv3 = self._bias_variable([64])


        # hidden layers
        conv1 = tf.nn.relu(tf.nn.conv2d(input_layer,
            W_conv1,
            strides = [1,4,4,1],
            padding = "VALID") + b_conv1)

        conv2 = tf.nn.relu(tf.nn.conv2d(conv1,
            W_conv2,
            strides = [1,2,2,1],
            padding = "VALID") + b_conv2)

        conv3 = tf.nn.relu(tf.nn.conv2d(conv2,
            W_conv3,
            strides = [1,1,1,1],
            padding = "VALID") + b_conv3)

        # Get shape of last unconnected layer
        conv3_shape = conv3.get_shape().as_list()
        shape = conv3_shape[1] * conv3_shape[2] * conv3_shape[3]

        W_fc1 = self._weight_variable([shape, 512])
        b_fc1 = self._bias_variable([512])

        W_fc2 = self._weight_variable([512, self._actions])
        b_fc2 = self._bias_variable([self._actions])

        # Flatten
        conv3_flat = tf.reshape(conv3,[-1,shape])

        # Fully Connected layers
        fc1 = tf.nn.relu(tf.matmul(conv3_flat,W_fc1) + b_fc1)

        Q_values = tf.matmul(fc1,W_fc2) + b_fc2

        # Using init model if 0
        if (which_model == 0):
            self._W_conv1 = W_conv1
            self._b_conv1 = b_conv1
            self._W_conv2 = W_conv2
            self._b_conv2 = b_conv2
            self._W_conv3 = W_conv3
            self._b_conv3 = b_conv3
            self._W_fc1 = W_fc1
            self._b_fc1 = b_fc1
            self._W_fc2 = W_fc2
            self._b_fc2 = b_fc2
        else:
            self._W_conv1T = W_conv1
            self._b_conv1T = b_conv1
            self._W_conv2T = W_conv2
            self._b_conv2T = b_conv2
            self._W_conv3T = W_conv3
            self._b_conv3T = b_conv3
            self._W_fc1T = W_fc1
            self._b_fc1T = b_fc1
            self._W_fc2T = W_fc2
            self._b_fc2T = b_fc2

        return Q_values, input_layer

    # Q Learning Function
    def _train(self):
        self._actionX = tf.placeholder("float", [None, self._actions], name = "actionX")
        self._y = tf.placeholder("float", [None], name = "y")

        Q_actions = tf.reduce_sum(tf.multiply(self._model,
            self._actionX), reduction_indices = 1)

        self._cost_function = tf.reduce_mean(tf.square(self._y - Q_actions))
        '''
        Performs Gradient Descent
        http://tflearn.org/optimizers/
        https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
        Possibly Apply Decay Learning Rate... Talk to Profs about it
        '''
        self._train = tf.train.RMSPropOptimizer(learning_rate = 0.00025,
            momentum=0.95,
            decay = 0.99,
            epsilon = 1e-6).minimize(self._cost_function)


    def _train_Qmatrix(self):
        new_envi_batch = []
        action_batch = []
        envi_batch = []
        reward_batch = []

        minibatch = random.sample(self._memory, MINI_BATCH_SIZE)

        for batch in minibatch:
            envi_batch.append(batch[0])
            action_batch.append(batch[1])
            reward_batch.append(batch[2])
            new_envi_batch.append(batch[3])

        y_batch = []
        Q_Val_batch = self._target_model.eval(feed_dict = {self._target_input_layer:new_envi_batch})

        # Grab best possible outcome
        for index in range(MINI_BATCH_SIZE):
            status = minibatch[index][4]
            if status:
                y_batch.append(reward_batch[index])
            else:
                y_batch.append(reward_batch[index] + (self._gamma * np.max(Q_Val_batch[index])))

        self._train.run(feed_dict = {self._y : y_batch,
                        self._actionX : action_batch,
                        self._input_layer : envi_batch})

        self._session.run(self._session_network)


    def remember(self, state):
        next_state = np.append(state[0], self._state[:,:,1:], axis = 2)
        # state, action, reward, next state, status
        self._memory.append((self._state, state[1], state[3], next_state, state[4]))
        if len(self._memory) >= MAX_MEM:
            self._memory = self._memory[1:]

        if self._frames_remembered > MINI_BATCH_SIZE:
            self._train_Qmatrix()

        self._frames_remembered += 1
        self._state = next_state


    def init_size(self, environment):
        self._state = np.stack((environment, environment, environment, environment), axis = 2)

    def take_action(self):
        array_len = len(self._memory) - (SIZE_QLEARNER + 1)
        # Feed it last X + current state
        if array_len > SIZE_QLEARNER:
            # Grab last X frames
            last_states = self._memory[:array_len:-1]
            states = [self._state]
            for i in range(len(last_states)):
                states.append(last_states[i][0])
            # Evaluate model and get Q Values
            q_value = self._model.eval(feed_dict = {self._input_layer : states})[0]
        else:
            # Evaluate model and get Q Values
            q_value = self._model.eval(feed_dict = {self._input_layer : [self._state]})[0]
        action_matrix = np.zeros(self._actions)
        index = 0

        # Do not start with high epsilon if loading model
        if (self._load):
            self._epsilon = self._epsilon_min
            self._load = False

        if random.random() <= self._epsilon:
            index = random.randrange(self._actions)
            action_matrix[index] = 1
        else:
            index = np.argmax(q_value)
            action_matrix[index] = 1

        if self._frames_remembered > TIMER:
            if self._epsilon > self._epsilon_min:
                self._epsilon *= self._epsilon_decay

        return action_matrix


    '''
    https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
    '''
    def process_image(self, observation):
        greyscale = cv2.cvtColor(cv2.resize(observation, (100, 100)), cv2.COLOR_BGR2GRAY)
        greyscale = greyscale[:84:]

        # Remove Borders
        contour = cv2.findContours(greyscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contour[0]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = greyscale[y:y+h,x:x+w]
        return np.reshape(crop,(len(crop),len(crop[0]),1))


    '''
    Code taken from TensorFlow Documentation
    https://www.tensorflow.org/get_started/mnist/pro
    '''
    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    '''
    Code taken from TensorFlow Documentation
    https://www.tensorflow.org/get_started/mnist/pro
    '''
    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)


    def save_model(self):
        path = self._saver.save(self._session, "model"+str(datetime.datetime.now()).split('.')[0])
        print("Model saved in file: %s" % path)


    '''
    https://www.tensorflow.org/programmers_guide/saved_model
    '''
    def load_model(self, path_to_meta):
        saver = tf.train.import_meta_graph(path_to_meta)
        saver.restore(self._session, tf.train.latest_checkpoint("./"))
        print("Model restored.")
        self._load = True


# Main Loop
def main(argv):
    # Get file to use to restore
    meta_to_restore = ''
    '''
    https://www.tutorialspoint.com/python/python_command_line_arguments.htm
    '''
    try:
        opts, args = getopt.getopt(argv, "m:")
    except getopt.GetoptError:
      print ('-m <metafile>')
      sys.exit(2)

    # Create game instance
    game_env = gym.make(GAME)

    # Create AI instance
    agent = custom_agent(game_env)

    for opt, arg in opts:
        if opt == "-m":
            meta_to_restore = arg
            print(meta_to_restore)
            agent.load_model(meta_to_restore)

    # Amount of games to play
    for instance in range(INSTANCES):
        total_frames = 0
        rewards = 0
        action = []

        environment = game_env.reset()
        # Shrink the image and greyscale it
        environment = agent.process_image(environment)
        agent.init_size(environment)

        agent._state = np.squeeze(agent._state)

        while True:
            game_env.render()

            if (total_frames % 4 == 0):
                action = agent.take_action()

            action_to_take = np.argmax(np.array(action))

            new_environment, reward, status, _ = game_env.step(action_to_take)

            # Shrink the image and greyscale it
            new_environment = agent.process_image(new_environment)

            agent.remember(np.array([new_environment, action, environment, reward, status]))

            if reward > 0:
            	rewards += 1
            if status:
                print("Instance Completed {}".format(instance))
                print("REWARDS %s" % rewards)
                break

            # Save tensor
        if (instance % 50 == 0):
            agent.save_model()
            print("Total Remembered Frames so far: %s" % agent._frames_remembered)


if __name__ == "__main__":
    main(sys.argv[1:])
