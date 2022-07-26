import argparse
import random
import tensorflow as tf
import os
import minerl
import gym
import convlstm_network
import numpy as np
from tensorflow.keras import datasets, layers, models
import tensorflow_probability as tfp
import random
import copy

#GLOBAL VARIABLES/PATHS
workspace_path= 'C:/Users/Halim/Downloads/minecraftRL/minecraft_bot_dev_A2C_final' # 'C:/Users/Halim/Downloads/minecraftRL/minecraft_bot_dev-python-five-frame-refactored-A2C/'
data_path='C:/Users/Halim/Downloads/minecraftRL/MineRLenv'
env_name = 'MineRLTreechop'
gpu_use = True

pretrained_model = True
pretrained_name = 'tree_supervised_model_9_1'
num_actions = 32

num_frames = 5

#MODEL SETUP
# Specify the height and width to which each video frame in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = num_frames #num_frames

NUM_ACTIONS = num_actions # 32 


turn_angle_left = -8 #LEFT (negative) camera turn #was all 20 before on seed 900 #-15 good
turn_angle_right = 8 #RIGHT (positive) camera turn #15 good
turn_angle_down = 8 #DOWN (positive) camera turn
turn_angle_up = -12 #UP (negative) camera turn #-20 good 12

class Model(tf.keras.Model):
    def __init__(self, IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH, NUM_ACTIONS):
        super().__init__()
        
        #MODEL SETUP
        # Specify the height and width to which each video frame in our dataset.
        self.image_height =  IMAGE_HEIGHT
        self.image_width = IMAGE_WIDTH

        # Specify the number of frames of a video that will be fed to the model as one sequence.
        self.sequence_length = SEQUENCE_LENGTH 
        self.num_actions = NUM_ACTIONS

        self.convlstm_1 = layers.ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                             recurrent_dropout=0.2, return_sequences=True, input_shape = (self.sequence_length,
                                                                                          self.image_height, self.image_width, 3)) #(5, 64, 64, 3)

        self.convlstm_2 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')
        self.timedrop_3 = layers.TimeDistributed(layers.Dropout(0.2))

        self.convlstm_4 = layers.ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                             recurrent_dropout=0.2, return_sequences=True)

        self.maxpol_5 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')
        self.timedrop_6 = layers.TimeDistributed(layers.Dropout(0.2))

        self.convlstm_7 = layers.ConvLSTM2D(filters = 14, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                             recurrent_dropout=0.2, return_sequences=True)

        self.maxpol_8 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')
        self.timedrop_9 = layers.TimeDistributed(layers.Dropout(0.2))

        self.timedrop_10 = layers.ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                             recurrent_dropout=0.2, return_sequences=True)

        self.maxpol_11 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last')
        
        self.flat_12 = layers.Flatten() #add couple of dense layers for each critic (100 or 50) and actor (100)
        
        self.actor_dense = layers.Dense(50, activation="relu", kernel_regularizer='l2')
        self.critic_dense = layers.Dense(50, activation="relu", kernel_regularizer='l2')

        #self.common = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')
        self.actor = layers.Dense(self.num_actions, kernel_regularizer='l2')
        self.critic = layers.Dense(1, kernel_regularizer='l2')
        #self.softmax_13 = layers.Dense(self.num_actions, activation = "softmax")

    def call(self, input_data):
        convlstm_1 = self.convlstm_1(input_data)
        convlstm_2 = self.convlstm_2(convlstm_1)
        timedrop_3 = self.timedrop_3(convlstm_2)
        convlstm_4 = self.convlstm_4(timedrop_3)
        maxpol_5 = self.maxpol_5(convlstm_4)
        timedrop_6 = self.timedrop_6(maxpol_5)
        convlstm_7 = self.convlstm_7(timedrop_6)
        maxpol_8 = self.maxpol_8(convlstm_7)
        timedrop_9 = self.timedrop_9(maxpol_8)
        timedrop_10 = self.timedrop_10(timedrop_9)
        maxpol_11 = self.maxpol_11(timedrop_10)
        flat_12 = self.flat_12(maxpol_11)
        actor_dense = self.actor_dense (flat_12)
        critic_dense = self.critic_dense (flat_12)
        return self.actor(actor_dense), self.critic(critic_dense)
#         softmax_13 = self.softmax_13(flat_12)
#         return softmax_13



def main():

    # GPU setup
    if gpu_use == True:
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_virtual_device_configuration(gpus[0],
        #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
        gpus = tf.config.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpus))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


     # Construct the required convlstm model.
    convlstm_model = Model(IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH, NUM_ACTIONS)
    

    # Load model weights
    if (pretrained_model == True):
        print("Load Pretrained Model")
        convlstm_model.load_weights("convlstm_model/" + pretrained_name)
    
    # Create the gym environment
    env = gym.make('MineRLTreechop-v0')

    seed = 358 #75 #980 #900 is walking forward 
    env.seed(seed)
    # tf.random.set_seed(seed)
    # np.random.seed(seed)

    for i_episode in range(0, 25000): #25000
        observation = env.reset()
        ten_frames = []
        new_lstm_list = []
        
        step = 0
        while True:
            step += 1
            state = observation['pov'] / 255.0 

            if step <= num_frames: #10
                # print(state.shape) #(64, 64, 3) <class 'numpy.ndarray'>
                ten_frames.append(state)
                # action_probs = 114
                if (step == num_frames): #10
                    new_lstm_list.append(copy.copy(ten_frames))
                    # print(new_lstm_list)
                    action_probs, critic_probs = convlstm_model.predict(tf.convert_to_tensor(new_lstm_list)) #tf.convert_to_tensor(new_lstm_list)
                    new_lstm_list = []
            else:
                # state_reformat = np.expand_dims(state, axis=0)
                # state_reformat2 = np.expand_dims(state_reformat, axis=0)
                ten_frames.pop(0)
                ten_frames.append(state)
                new_lstm_list.append(copy.copy(ten_frames))
                # print(new_lstm_list)

                action_probs, critic_probs = convlstm_model.predict(tf.convert_to_tensor(new_lstm_list)) #tf.convert_to_tensor(new_lstm_list)
                new_lstm_list = []

                
            if random.random() <= 0.05:
                action_index = random.randint(0,num_actions-1)
                # action_index=115
            else:
                if step < num_frames : #10
                    action_index=22
                else:
                    action_index= np.argmax(action_probs[0]) #PREDICTION
            print(action_index)
            # uses a distribution to find the action number
            # tfd = tfp.distributions
            # action_dist = tfd.Categorical(probs=action_probs)
            # action_index = int(action_dist.sample()[0])

            action = env.action_space.noop()

            if (action_index == 0): #left (negative) camera turn
                action['camera'] = [0, turn_angle_left]
                action['forward'] = 1
                action['attack'] = 1
            elif (action_index == 1):
                action['camera'] = [0, turn_angle_left]
                action['forward'] = 1
                action['jump'] = 1
            elif (action_index == 2):
                action['camera'] = [0, turn_angle_left]
                action['forward'] = 1
            elif (action_index == 3):
                action['camera'] = [0, turn_angle_left]
                action['attack'] = 1
            elif (action_index == 4):
                action['camera'] = [0, turn_angle_left]


            elif (action_index == 5): #right (positive) camera turn
                action['camera'] = [0, turn_angle_right]
                action['forward'] = 1
                action['attack'] = 1
            elif (action_index == 6):
                action['camera'] = [0, turn_angle_right]
                action['forward'] = 1
                action['jump'] = 1
            elif (action_index == 7):
                action['camera'] = [0, turn_angle_right]
                action['forward'] = 1
            elif (action_index == 8):
                action['camera'] = [0, turn_angle_right]
                action['attack'] = 1
            elif (action_index == 9):
                action['camera'] = [0, turn_angle_right]


            elif (action_index == 10): #up (negative) camera turn 
                action['camera'] = [turn_angle_up, 0]
                action['forward'] = 1
                action['attack'] = 1
            elif (action_index == 11):
                action['camera'] = [turn_angle_up, 0]
                action['forward'] = 1
                action['jump'] = 1
            elif (action_index == 12):
                action['camera'] = [turn_angle_up, 0]
                action['forward'] = 1
            elif (action_index == 13):
                action['camera'] = [turn_angle_up, 0]
                action['attack'] = 1
            elif (action_index == 14):
                action['camera'] = [turn_angle_up, 0]    


            elif (action_index == 15): #down (positive) camera turn 
                action['camera'] = [turn_angle_down, 0]
                action['forward'] = 1
                action['attack'] = 1
            elif (action_index == 16):
                action['camera'] = [turn_angle_down, 0]
                action['forward'] = 1
                action['jump'] = 1
            elif (action_index == 17):
                action['camera'] = [turn_angle_down, 0]
                action['forward'] = 1
            elif (action_index == 18):
                action['camera'] = [turn_angle_down, 0]
                action['attack'] = 1
            elif (action_index == 19):
                action['camera'] = [turn_angle_down, 0]  


            elif (action_index == 20): #no camera turn
                action['forward'] = 1
                action['attack'] = 1
            elif (action_index == 21):
                action['forward'] = 1
                action['jump'] = 1
            elif (action_index == 22):
                action['forward'] = 1

            elif (action_index == 23):
                action['back'] = 1
                action['attack'] = 1
            elif (action_index == 24): 
                action['back'] = 1


            elif (action_index == 25):
                action['right'] = 1
                action['attack'] = 1
            elif (action_index == 26):
                action['right'] = 1


            elif (action_index == 27):
                action['left'] = 1
                action['attack'] = 1
            elif (action_index == 28):
                action['left'] = 1


            elif (action_index == 29):
                action['attack'] = 1

            elif (action_index == 30): 
                action['jump'] = 1

            elif (action_index == 31):
                pass 

            observation, reward, done, info = env.step(action)
            env.render()

            if done:
                print("Total step: {:.2f}".format(step))
                step = 0
                break

    env.close()


if __name__ == '__main__':
    main()