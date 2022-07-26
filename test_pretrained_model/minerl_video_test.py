import argparse
import random
import tensorflow as tf
import os
import minerl
import numpy as np
import copy
import datetime
from tensorflow.keras import datasets, layers, models



#GLOBAL VARIABLES/PATHS
workspace_path= 'C:/Users/Halim/Downloads/minecraftRL/canada_compute_backup_final/minecraft_bot_dev_A2C_final_100neuron_dense/test_pretrained_model'#'C:/Users/Halim/Downloads/minecraftRL/canada_compute_backup_final/minecraft_bot_dev_A2C_final/test_pretrained_model/convlstm_model'
data_path= 'C:/Users/Halim/Downloads/minecraftRL/MineRLenv' #'/home/endeavor/projects/def-mcrowley/endeavor'
env_name = 'MineRLTreechop'
gpu_use = True

pretrained_model = True
pretrained_name = 'tree_supervised_model_0_31' #step 32 from tensorboard
num_epochs = 1 #epochs per video #4
iter_all_eps = 1 #number of times train on all 15 testing videos/episodes
num_actions = 32
num_frames = 5



#MODEL SETUP
# Specify the height and width to which each video frame in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = num_frames #num_frames

NUM_ACTIONS = num_actions # 32 

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
        
        self.actor_dense = layers.Dense(100, activation="relu", kernel_regularizer='l2')
        self.critic_dense = layers.Dense(100, activation="relu", kernel_regularizer='l2')

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

def discounted_rewards (rewards, gamma):
    discnt_rewards = []
    action_np_list = []
    sum_reward = 0
    rewards.reverse()
    for r in rewards:
      sum_reward = r + gamma*sum_reward
      discnt_rewards.append(sum_reward)
    discnt_rewards.reverse()
    discnt_rewards = np.array(discnt_rewards, dtype=np.float32)

    return discnt_rewards

def main():

    tree_data = minerl.data.make('MineRLTreechop-v0', data_dir=data_path)

    #GPUS SETUP
    if gpu_use == True:
        gpus = tf.config.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpus))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    ###############################################################
    # MODEL SETUP START
    #location for model summary %tensorboard --logdir logs/gradient_tape
    count = 0
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    test_log_dir = workspace_path + '/logs/gradient_tape/' + current_time + '/train'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Construct the required convlstm model.
    convlstm_model = Model(IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH, NUM_ACTIONS)
    
    #FOR RERUN WITH EXISTING MODEL SET TO TRUE AND CHANGE NAME
    if (pretrained_model == True):
        print("Load Pretrained Model")
        convlstm_model.load_weights("convlstm_model/" + pretrained_name)
    
    # Prepare the metrics.
    actor_test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy('actor_test_accuracy')
    critic_test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy('critic_test_accuracy') #Fixx this
    val_accuracy_metric = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)
    # test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
    
    # MODEL SETUP END
    ###############################################################
    # DATA PROCESSING START
    trajectory_names = tree_data.get_trajectory_names()
    num_trajectories = len(trajectory_names)
    print("len(trajectory_names): ", len(trajectory_names))

    for iterate_all_train_episodes in range(0, iter_all_eps):
        for training_episode in range(0, num_trajectories): 

            trajectory_name = trajectory_names[training_episode]
            print("trajectory_name: ", trajectory_name)
                
            trajectory = tree_data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
                
            noop_action_num = 0
                
            all_actions = []
            all_obs = []
            all_rewards = []
            train_loss=[]
            train_acc=[]
            for dataset_observation, dataset_action, reward, next_state, done in trajectory:  
     
                observation = dataset_observation['pov'] / 255.0

                act_cam_0 = dataset_action['camera'][0]
                act_cam_1 = dataset_action['camera'][1]
                act_attack = dataset_action['attack'] #1
                act_forward = dataset_action['forward'] #2
                act_jump = dataset_action['jump'] #3
                act_back = dataset_action['back'] #4
                act_left = dataset_action['left'] #5
                act_right = dataset_action['right'] #6
                act_sneak = dataset_action['sneak'] #7

                if (abs(act_cam_0 > 0) or abs(act_cam_1 > 0)): #if there was a change in camera at all
                    if ( (act_cam_1 < 0) & ( abs(act_cam_0) < abs(act_cam_1) ) ): #left (negative) camera turn
                        
                        if (act_forward == 1):
                            if (act_attack == 1): # 'turn_angle_left','forward','attack'
                                act_index = 0
                            elif (act_jump == 1): # 'turn_angle_left','forward','jump'
                                act_index = 1
                            else:
                                act_index = 2 # 'turn_angle_left','forward'
                        elif (act_attack == 1): # 'turn_angle_left','attack'
                            act_index = 3
                        else:
                            act_index = 4 # 'turn_angle_left'


                    elif ( (act_cam_1 > 0) & ( abs(act_cam_0) < abs(act_cam_1) ) ): #right (positive) camera turn
                        
                        if (act_forward == 1):
                            if (act_attack == 1): # 'turn_angle_right','forward', 'attack'
                                act_index = 5
                            elif (act_jump == 1): # 'turn_angle_right','forward','jump'
                                act_index = 6
                            else:
                                act_index = 7 # 'turn_angle_right', 'forward'
                        elif (act_attack == 1): # 'turn_angle_right','attack'
                            act_index = 8
                        else:
                            act_index = 9 # 'turn_angle_right'
                        

                    elif ( (act_cam_0 < 0) & ( abs(act_cam_0) > abs(act_cam_1) ) ): #up (negative) camera turn #now up
                        
                        if (act_forward == 1):
                            if (act_attack == 1): # 'turn_angle_up','forward','attack'
                                act_index = 10
                            elif (act_jump == 1): # 'turn_angle_up','forward','jump'
                                act_index = 11
                            else:
                                act_index = 12 # 'turn_angle_up','forward'
                        elif (act_attack == 1): # 'turn_angle_up', 'attack'
                            act_index = 13
                        else:
                            act_index = 14 # 'turn_angle_up'

                    elif ( (act_cam_0 > 0) & ( abs(act_cam_0) > abs(act_cam_1) ) ): #down (positive) camera turn #now down
                        
                        if (act_forward == 1):
                            if (act_attack == 1): # 'turn_angle_down','forward','attack'
                                act_index = 15
                            elif (act_jump == 1): # 'turn_angle_down','forward','jump'
                                act_index = 16
                            else:
                                act_index = 17 # 'turn_angle_down','forward'
                        elif (act_attack == 1): # 'turn_angle_down','attack'
                            act_index = 18
                        else:
                            act_index = 19 # 'turn_angle_down'


                else: #no camera movement
                    if (act_forward == 1):
                        if (act_attack == 1): # 'forward','attack'
                            act_index = 20
                        elif (act_jump == 1): # 'forward','jump'
                            act_index = 21
                        else:                 # 'forward'
                            act_index = 22

                    elif (act_back == 1):
                        if (act_attack == 1): # 'back','attack'
                            act_index = 23
                        else:                 # 'back' (additional)
                            act_index = 24

                    elif (act_right == 1): 
                        if (act_attack == 1): # 'right','attack'
                            act_index = 25
                        else:
                            act_index = 26    # 'right'

                    elif (act_left == 1):
                        if (act_attack == 1): # 'left','attack'
                            act_index = 27
                        else:                 # 'left'
                            act_index = 28

                    elif (act_attack == 1): # 'attack'
                        act_index = 29

                    elif (act_jump == 1): # 'jump' (additional)
                        act_index = 30

                    else:                 # 'pass'/'no movement'
                        act_index = 31
                    

                all_obs.append(observation)#removed [observation]
                all_actions.append(np.array([act_index]))
                all_rewards.append(reward)

            print(f"Video Counter = {training_episode+1}")
            batch = tuple()
            batch = (all_obs, all_actions, all_rewards)
        
            episode_size = len(batch[0]) #only 1 batch per video
            print("episode_size: ", episode_size) #number of images in the batch/video

            replay_obs_list = batch[0] #all images in the single batch/video
            replay_act_list = batch[1] #all actions per image in the single batch/video
            replay_reward_list = batch[2]

            discnt_rewards = discounted_rewards(replay_reward_list, 1)

#             replay_obs_lstm = frame_partition(replay_obs_list)
            for current_epoch_vid in range (0, num_epochs):
                for episode_index in range(0, episode_size): # range(0, episode_size, SEQUENCE_LENGTH)
                    print("iterate_all_train_episodes: ", iterate_all_train_episodes+1)
                    print("current_episode_epoch: ", current_epoch_vid+1)
                    print("episode_index: ", episode_index)
                    
                    replay_obs_list = np.array(replay_obs_list)
                    replay_act_list = np.array(replay_act_list)
                    discnt_replay_reward_list = np.array(discnt_rewards) # was replay_reward_list
                    obs = replay_obs_list[episode_index:episode_index+SEQUENCE_LENGTH,:,:,:] #(2, 64, 64, 3)
                    act = replay_act_list[episode_index:episode_index+SEQUENCE_LENGTH,:] #(2)
                    reward = discnt_replay_reward_list [episode_index:episode_index+SEQUENCE_LENGTH] #(2)

                    if len(obs) != SEQUENCE_LENGTH: #len(obs) = current frames
                        break
                    
                    replay_act_array = tf.concat(act[SEQUENCE_LENGTH-1], 0) #https://www.tensorflow.org/api_docs/python/tf/concat
                    replay_reward_array = tf.concat(reward[SEQUENCE_LENGTH-1], 0)

                    # print(f"act: {act}")
                    # print(f"replay_act_array: {replay_act_array}")

                    # print(f"reward: {reward}")
                    # print(f"replay_reward_array: {replay_reward_array}")

                    batch_size = len(obs)
                    tf.print("batch_size: ", batch_size)
                    
                    replay_act_array_onehot = tf.one_hot(replay_act_array, num_actions)
                    # print(f"replay_act_array_onehot type: {type(replay_act_array_onehot)}")
                    # print(f"replay_act_array_onehot: {replay_act_array_onehot}")

                    reward_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
                    reward_array = reward_array.write(0, replay_reward_array)
                    reward_array = reward_array.stack()
                    print(f"reward_array type: {type(reward_array)}")
                    print(f"reward_array: {reward_array}")

        # DATA PROCESSING END
        ###############################################################
        # MODEL TESTING START

                    prediction, value = convlstm_model(tf.expand_dims(obs,0)) 
                    # print(f"prediction type: {type(prediction)}")
                    # print(f"prediction: {prediction}")
                    # print(f"value type: {type(value)}")
                    # print(f"value: {value}")

                    v = tf.reshape(value, (len(value),)) #critic (Q) value
                    # td = tf.math.subtract(replay_reward_array, v) #advantage/temporal difference
                    print(f"v type: {type(v)}")
                    print(f"v: {v}")

                    # act_probs = act_probs.write(0, prediction[0]) 
                    # act_probs = act_probs.stack()
                    # print(f"act_probs type: {type(act_probs)}")
                    # print(f"act_probs: {act_probs}")

                    # Update training metric.###################HERE#######################
                    # act_probs_onehot = tf.one_hot(tf.math.argmax(act_probs,1), num_actions) 
                    actor_test_accuracy_metric.update_state(replay_act_array_onehot, prediction) # or act_probs or act_probs_onehot
                    critic_test_accuracy_metric.update_state(reward_array, v)
                    #^within current x frames; looping through the frames of a episode/video##################################
                
                # Display metrics at the end of each epoch.
                actor_test_acc = actor_test_accuracy_metric.result()
                print("Training acc over epoch: %.4f" % (float(actor_test_acc),))
                critic_test_acc = critic_test_accuracy_metric.result()
                print("Training acc over epoch: %.4f" % (float(critic_test_acc),))

                #location for model summary
                if (current_epoch_vid+1 == num_epochs): #after training on a video (after all epochs per vid) it saves loss and accuracy
                    count = count+1
                    print(f"Counter Total= {count}")
                    with test_summary_writer.as_default(): 
                        tf.summary.scalar('actor_accuracy', actor_test_accuracy_metric.result(), step=count)
                        tf.summary.scalar('critic_accuracy', critic_test_accuracy_metric.result(), step=count)

                # Reset training metrics at the end of each epoch
                actor_test_accuracy_metric.reset_states()
                critic_test_accuracy_metric.reset_states()
                
                print("")



if __name__ == '__main__':
    main()
