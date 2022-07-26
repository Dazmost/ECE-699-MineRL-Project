import argparse
import random
import tensorflow as tf
import os
import minerl
import numpy as np
import copy
import datetime
from tensorflow.keras import datasets, layers, models
import gym
import tensorflow_probability as tfp


#GLOBAL VARIABLES/PATHS
workspace_path= 'C:/Users/Halim/Downloads/minecraftRL/minecraft_bot_dev_A2C_final'
data_path= 'C:/Users/Halim/Downloads/minecraftRL/MineRLenv'
env_name = 'MineRLTreechop'
gpu_use = True

pretrained_model = True
pretrained_name = 'tree_supervised_model_9_1' # 20 frame model 'tree_supervised_model_0_11'
num_epochs = 2 #epochs per video #4
iter_all_eps = 100 #number of times train on all 82 training videos/episodes
num_actions = 32
num_frames = 5

turn_angle_left = -8 #LEFT (negative) camera turn #was all 20 before on seed 900 #-15 good
turn_angle_right = 8 #RIGHT (positive) camera turn #15 good
turn_angle_down = 8 #DOWN (positive) camera turn
turn_angle_up = -12 #UP (negative) camera turn #-20 good 12

# Prepare the metrics.
actor_loss_metric = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
critic_loss_metric = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = workspace_path + '/A2C_logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# count = 0

#MODEL SETUP
# Specify the height and width to which each video frame in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = num_frames #num_frames

NUM_ACTIONS = num_actions 

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

class agent():
    def __init__(self, gamma = 0.99, pretrained_model = False):
        self.gamma = gamma
        self.count = 0
        self.a_opt = tf.keras.optimizers.Adam(0.0005)
        # self.c_opt = tf.keras.optimizers.Adam(0.0005)
        self.actor_critic = Model(IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH, NUM_ACTIONS)
        # Load model weights
        if (pretrained_model == True):
            print("Load Pretrained Model")
            self.actor_critic.load_weights("convlstm_model/" + pretrained_name)

    
    def act(self,state):
        logits, value = self.actor_critic(tf.convert_to_tensor(state))
        #prob = tf.nn.softmax(logit)
        #prob = prob.numpy()
        #dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        dist = tfp.distributions.Categorical(logits=logits, dtype=tf.float32)
        action = dist.sample() #Note that a call to sample() without arguments will generate a single sample.
        return int(action.numpy()[0])
    
    #input actions and policy probability over the step batch 
    def actor_loss(self, probs, action, td): #softmax probs (new prediction in grad_tape), action_lists (int of old prediction), advantage/temporal difference
        
        # determines probability and log probability of each action in actions (list)
        # probability = []
        # log_probability= []
        # for pb, act in zip(probs,actions):
        dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        prob = dist.prob(action)
        # probability.append(prob)
        # log_probability.append(log_prob)

        # p_loss= []
        # e_loss = []
        td = td.numpy()
        # for pb, t, lpb in zip(probability, td, log_probability):
        td =  tf.constant(td)
        policy_loss = tf.math.multiply(log_prob,td) #multiply log probability with advantage/temporal difference
        entropy_loss = tf.math.negative(tf.math.multiply(prob,log_prob)) #negative of probability and log probability 
        # p_loss.append(policy_loss)
        # e_loss.append(entropy_loss)
        # p_loss = tf.stack(p_loss)
        # e_loss = tf.stack(e_loss)
        # p_loss = tf.reduce_mean(p_loss) #Computes the mean of elements across dimensions of a tensor
        # e_loss = tf.reduce_mean(e_loss)
    

        # policy_loss * ACTOR_LOSS_WEIGHT - entropy_loss * ENTROPY_LOSS_WEIGHT
        # loss = -p_loss - 0.0001 * e_loss
        loss = -policy_loss - 0.0001 * entropy_loss
        #print(loss)
        return loss

    #train every step batch
    def learn(self, states, actions, discnt_rewards): # states_list, actions_list, rewards_list
        # print(states)
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        #tf.convert_to_tensor(state)

        # for state, action, discnt_reward in zip(states, actions, discnt_rewards): ##########################################
        for episode_index in range(0, len(states)): # range(0, episode_size, SEQUENCE_LENGTH)
            print("HERE2")
            print(len(states)) #20
            print(len(states[0])) # 64   # was 1 or num length
            print(len(states[0][0])) # 64
            print(len(states[0][0][0])) # 3  # was 64
            # print(len(states[0][0][0][0])) #3
            print("HERE2")
            obs_list = np.array(copy.deepcopy(states))
            act_list = np.array(actions)
            discnt_list = np.array(discnt_rewards)
            # print(obs_list[0])
            # print(obs_list[1])
            obs = obs_list[episode_index:episode_index+SEQUENCE_LENGTH,:,:,:] #(2, 64, 64, 3)
            # act = act_list[episode_index:episode_index+SEQUENCE_LENGTH,:]
            act = act_list[episode_index:episode_index+SEQUENCE_LENGTH]
            reward = discnt_list [episode_index:episode_index+SEQUENCE_LENGTH] #(2)

            if len(obs) != SEQUENCE_LENGTH: #len(obs) = current frames
                break
            
            action_value = tf.concat(act[SEQUENCE_LENGTH-1], 0) #https://www.tensorflow.org/api_docs/python/tf/concat
            reward_value = tf.concat(reward[SEQUENCE_LENGTH-1], 0)
            # batch_size = len(obs)
            # tf.print("batch_size: ", batch_size)
            

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                logits, value = self.actor_critic(tf.expand_dims(obs,0), training=True) #new prediction for state
                p = tf.nn.softmax(logits)
                
                v = tf.reshape(value, (len(value),)) #critic (Q) value
                td = tf.math.subtract(reward_value, v) #advantage/temporal difference

                a_loss = self.actor_loss(p, action_value, td)
                #keras.losses.mean_squared_error(discounted_rewards, predicted_values) * CRITIC_LOSS_WEIGHT
                c_loss = 0.5*tf.keras.losses.mean_squared_error(reward_value, v)
                
            grads1 = tape1.gradient([a_loss, c_loss], self.actor_critic.trainable_variables)
            # grads2 = tape2.gradient(c_loss, self.actor_critic.trainable_weights)
            self.a_opt.apply_gradients(zip(grads1, self.actor_critic.trainable_variables))
            # self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
            actor_loss_metric.update_state(a_loss)
            critic_loss_metric.update_state(c_loss)

        #location for model summary
        self.count = self.count+1
        print(f"Step batch total= {self.count}")
        with train_summary_writer.as_default(): #after step_size it saves loss and accuracy
            tf.summary.scalar('actor_loss', actor_loss_metric.result(), step=self.count)
            tf.summary.scalar('critic_loss', critic_loss_metric.result(), step=self.count)

        #  if (training_episode % 1 == 0) and (current_epoch_vid+1 == num_epochs):#100 # saves after every episode/train video (num_epochs)
        #     print("Model Saved!")
        #     convlstm_model.save_weights(workspace_path + '/convlstm_model/tree_supervised_model_' + str(iterate_all_train_episodes)+ '_'  + str(training_episode))
        
        return actor_loss_metric.result(), critic_loss_metric.result() #a_loss and c_loss

#https://stackoverflow.com/questions/59690188/how-do-i-make-a-multi-output-tensorflow-2-custom-training-loop-for-both-regressi

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
    
    # GPU setup
    if gpu_use == True:
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_virtual_device_configuration(gpus[0],
        #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
        gpus = tf.config.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpus))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Create the gym environment
    env = gym.make('MineRLTreechop-v0')
    seed = 900 #75 #980 #900 is walking forward 
    env.seed(seed)
    # tf.random.set_seed(seed)
    # np.random.seed(seed)
    
    # Construct the required convlstm model.
    agentoo7 = agent(pretrained_model = pretrained_model)
    steps = 25000
    ep_reward = []
    total_avgr = []
    
    for i_episode in range(0, steps): #25000
        
        done = False
        observation = env.reset()
        total_reward = 0
        all_aloss = []
        all_closs = []
        rewards_list = []
        states_list = []
        actions_list = []
        
        ten_frames = []
        new_lstm_list = []
        
        
        step = 0
        while not done:
            env.render()
            step += 1
            state = observation['pov'] / 255.0 

            if step <= num_frames: #10
                # print(state.shape) #(64, 64, 3) <class 'numpy.ndarray'>
                ten_frames.append(state)
                # action_probs = 114
                if (step == num_frames): #10
                    new_lstm_list = []
                    new_lstm_list.append(copy.deepcopy(ten_frames))
                    # print(new_lstm_list)
                    #action_probs = convlstm_model.predict(tf.convert_to_tensor(new_lstm_list)) #tf.convert_to_tensor(new_lstm_list)
                    action_sample = agentoo7.act(new_lstm_list)
            else:
                # state_reformat = np.expand_dims(state, axis=0)
                # state_reformat2 = np.expand_dims(state_reformat, axis=0)
                
                #old state
#                 old_lstm_list = []
#                 old_lstm_list.append(copy.copy(ten_frames))
                
#                 aloss, closs = agentoo7.learn(old_lstm_list, action_index, reward, new_lstm_list, done)
#                 all_aloss.append(aloss)
#                 all_closs.append(closs)

                # ten frame TOO MUCH MEMORY
                # print(f"Length of ten_frames: {len(ten_frames)} and {len(ten_frames[0])} and {len(ten_frames[0][0])} and {len(ten_frames[0][0][0])}")
                # states_list.append(copy.deepcopy(ten_frames)) #state https://docs.python.org/3/library/copy.html
                
                states_list.append(copy.deepcopy(state))

                #new state
                ten_frames.pop(0)
                ten_frames.append(state)
                new_lstm_list = []
                new_lstm_list.append(copy.deepcopy(ten_frames))
                # print(new_lstm_list)

                #action_probs = convlstm_model.predict(tf.convert_to_tensor(new_lstm_list)) #tf.convert_to_tensor(new_lstm_list)
                action_sample = agentoo7.act(new_lstm_list)              
                
            if random.random() <= 0.05:
                action_index = random.randint(0,114)
                # action_index=115
            else:
                if step < num_frames : #10
                    action_index=114
                else:
                    action_index = action_sample #PREDICTION
                    #action_index = np.argmax(action_probs[0])
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
            rewards_list.append(reward)
            # ten frame
            # print(f"Length of ten_frames: {len(ten_frames)} and {len(ten_frames[0])} and {len(ten_frames[0][0])} and {len(ten_frames[0][0][0])}")
            # states_list.append(copy.deepcopy(ten_frames)) #state https://docs.python.org/3/library/copy.html
            # print(states_list)
            # new_lstm_list
            # print(f"Length of new_lstm_list: {len(new_lstm_list)} and {len(new_lstm_list[0])} and {len(new_lstm_list[0][0])} and {len(new_lstm_list[0][0][0])}")
            # states_list.append(copy.deepcopy(new_lstm_list)) #state https://docs.python.org/3/library/copy.html

            #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
            actions_list.append(action_index)
            total_reward += reward
            print("total_reward:", total_reward)
            
            if (step%20 == 0) and (step !=0): #20-num_frames
                # done = True
                discnt_rewards = discounted_rewards(rewards_list, 1)
                print(f"State length: {len(states_list)}")
                print(f"Action length: {(actions_list)}")
                print(f"Rewards length: {(discnt_rewards)}")
                print("HERE")
                # print(type(states_list))
                print(len(states_list)) # 20
                # print(type(states_list[0]))
                print(len(states_list[0])) # 1 should be num_frames =5

                print(len(states_list[0][0])) # 64
                print(len(states_list[0][0][0])) # 64
                # print(len(states_list[0][0][0][0])) # should be 3 # dont need anymore
                print("HERE")
                al,cl = agentoo7.learn(states_list, actions_list, discnt_rewards) 
                states_list = []
                rewards_list = []
                actions_list = []
                print(f"actor_loss = {al} for step {step}") 
                print(f"critic_loss = {cl} for step {step}")

            if done:
                ep_reward.append(total_reward) #outside step loop/episode
                avg_reward = np.mean(ep_reward[-100:])
                total_avgr.append(avg_reward)
                print("total reward after {} steps is {} and avg reward is {}".format(step, total_reward, avg_reward))
                # states_process, actions_process, discnt_rewards = discounted_rewards(states_list, actions_list, rewards, 1)

                # al,cl = agentoo7.learn(states_process, actions_process, discnt_rewards) 
                # print(f"al{al}") 
                # print(f"cl{cl}")      
      
    
            #if done:
#                 print("Total step: {:.2f}".format(step))
                #print("total reward after {} steps is {}".format(step, total_reward))
                #step = 0
                #break

    env.close()


if __name__ == '__main__':
    main()