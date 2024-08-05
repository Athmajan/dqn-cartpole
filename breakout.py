import numpy as np
import gym
np.bool8 = np.bool

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from dataclasses import dataclass
from typing import Any
from random import sample, random


import wandb
from tqdm import tqdm
from collections import deque
from models import ConvModel, Model


from utils import FrameStackingAndResizingEnv
from configs import lisstofSets

'''
Tutorial https://www.youtube.com/watch?v=WHRQUZrxxGw&t=1163s&ab_channel=JackofSome

'''
# The replay buffer is nothing but a collection of
# the sequences of STATE, ACTION, REWARD and NEXT STATE
# Therefore it is easier to create a dataclass for this 
# for better organization
@dataclass
class Sarsd:
    state: Any
    action : int
    reward : float
    next_state : Any
    done : bool

class DQNAgent:
    def __init__(self,model):
        self.model = model
    def get_actions(self,observations):
        # since observation contains velocity
        # already we do not need to stack obs.
        # obs_shape is (N, 4) N being batch size
        q_vals = self.model(observations)

        # q_vals shape (N, 2) q values for 2 action choces.
        return q_vals.max(-1)[1]
   
  

# Improve this with python deque
# also can be a database?
# deque can be dropped and made faster
class ReplayBuffer:
    def __init__(self,buffer_size = 100000):
        self.buffer_size = buffer_size
        self.buffer = [None]*buffer_size # fixed size array
        self.idx = 0

    def insert(self,sars):
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_samples):
        
        assert num_samples < min(self.idx,self.buffer_size)

        if self.idx < self.buffer_size:
        # until we reach the buffer size we cant  sample
        # from the entire array but sample upto idx only
            return sample(self.buffer[:self.idx],num_samples)
        return sample(self.buffer,num_samples)
            

        # return sample(self.buffer, num_samples)



def update_tgt_model(m,tgt):
    '''
    THis is to copy the weights from one to another
    '''
    tgt.load_state_dict(m.state_dict())

def train_step(model,state_transitions,tgt,num_actions, gamma=0.99):
        # need to create the state vector
        # that is a stacked

        cur_states = torch.stack([torch.Tensor(s.state) for s in state_transitions])
        rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions])
        
        # we need to make the future rewards to zero
        # if the episode is done. So 1 if weren't done
        # 0 if were done
        mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])

        next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions])
        actions = [s.action for s in state_transitions]


        with torch.no_grad():
            # we dont do backprop on target model
            qval_next = tgt(next_states).max(-1)[0] # max of soemthig shaped (N, num_actions)

        model.opt.zero_grad()
       
        # we need to pick only the q values for the actions that we chose
        qvals = model(cur_states) # (N, num_actions)
        one_hot_actions = F.one_hot(torch.LongTensor(actions),num_actions)
        
        
        # ignoring the discount factor for now

        # check deep rl tutorial david silver for this refrence 
        # loss = ((rewards +  mask[:,0]*qval_next*gamma - torch.sum(qvals*one_hot_actions,-1))**2).mean()

        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(torch.sum(qvals*one_hot_actions,-1),rewards.squeeze() +  mask[:,0]*qval_next*gamma)
        loss.backward()
        
        model.opt.step()

        return loss

def run_test_episode(model,env,max_steps=1000): # reward, movie?
    frames = []

    obs = env.reset()
    frames.append(env.frame)
    frame = env.frame
    
    idx = 0
    done = False
    reward = 0
    while not done and idx < max_steps :
        action = model(torch.Tensor(obs).unsqueeze(0)).max(-1)[-1].item()
        observation, r, done, _ = env.step(action)
        reward += r
        frames.append(env.frame) 

    return reward, np.stack(frames,0)

def main(name,test=False, chkpt=None):

    params  = lisstofSets[name]


    # if not test:
    #     wandb.init(project="dqn-breakout",name=name)
    done = False
    do_boltzmann_explore = False
    
    min_rb_size = params["min_rb_size"]
    sample_size =  params["sample_size"]
    

    eps_decay =  params["eps_decay"]

    env_steps_before_train = params["env_steps_before_train"]
    tgt_model_update = params["tgt_model_update"]
    epoch_before_test = 500

    # done  = False
   

    np.bool8 = np.bool
    env = gym.make("Breakout-v4")
    env = FrameStackingAndResizingEnv(env,84,84,4)

    test_env = gym.make("Breakout-v4")
    test_env = FrameStackingAndResizingEnv(test_env,84,84,4)


    # 
    last_observation = env.reset() 
    
    m = ConvModel(env.observation_space.shape,env.action_space.n) # model that we train
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))

    # target model to update. Check David Silver lecture
    # for background information
    tgt = ConvModel(env.observation_space.shape,env.action_space.n) # fixed model
    update_tgt_model(m,tgt)

        
    rb = ReplayBuffer()
    steps_since_train = 0
    epochs_since_tgt = 0
    epochs_since_test = 0
    
    steps_num = -1 * min_rb_size
    # qvals = m(torch.Tensor(last_observation))


    episode_rewards = []
    rolling_reward = 0
    tq = tqdm()
    try:
        while True:
            if test:
                env.render()
                time.sleep(0.05)
            tq.update(1)

            eps =eps_decay**(steps_num)

            if test:
                eps = 0
            if do_boltzmann_explore:
                logits = m(torch.Tensor(last_observation).unsqueeze(0))[0]
                action = torch.distributions.Categorical(logits=logits).sample().item()

            else:
                if random() < eps:
                    # use completely random action
                    action = env.action_space.sample()
                else:
                    # use the agent to get the action
                    action = m(torch.Tensor(last_observation).unsqueeze(0)).max(-1)[-1].item()

            # action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            rolling_reward += reward

            # reward = reward*0.01

            rb.insert(Sarsd(last_observation, action,reward,observation,done))
            last_observation = observation  

            if done:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                rolling_reward = 0

                observation = env.reset()

            steps_since_train += 1
            steps_num += 1
            
            if (not test) and rb.idx > min_rb_size and steps_since_train > env_steps_before_train:
                
                # it can be seen that the replay is duplicated
                # but this keeps the design of the the replay buffer
                # simple at this point. think of other RBs to improve
                loss=  train_step(m,rb.sample(sample_size),tgt,env.action_space.n)

                # if not np.isnan(np.mean(episode_rewards)):
                #     wandb.log({'loss':loss.detach().item(), 'eps':eps, 'avg_reward' : np.mean(episode_rewards)},step=steps_num) 
                
                episode_rewards = []
                epochs_since_tgt += 1
                epochs_since_test += 1

                # if epochs_since_test > epoch_before_test:
                #     rew, frames= run_test_episode(m,test_env)
                #     # frames.shape == (Time, Height, Width, Channels)
                #     wandb.log({'test_reward':rew, 'test_video':wandb.Video(frames.transpose(0,3,1,2),str(rew),fps=30,format="mp4")})
                #     epochs_since_test = 0

                if epochs_since_tgt > tgt_model_update:
                    print("Updating the Target Model")
                    update_tgt_model(m,tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(),f"breakout_models/{steps_num}.pth")

                steps_since_train = 0

                

    except KeyboardInterrupt:
        # wandb.finish()
        pass

    env.close()
    # wandb.finish()




 
if __name__ == '__main__':
    main("setting_cartPole_2_nofire",False)
    # argh.dispatch_command(main)