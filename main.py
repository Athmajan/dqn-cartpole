import gymnasium as gym
import numpy as np
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
from models import Model


from utils import FrameStackingAndResizingEnv

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
        # import ipdb; ipdb.set_trace()
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
        loss = ((rewards +  mask[:,0]*qval_next*gamma - torch.sum(qvals*one_hot_actions,-1))**2).mean()
        loss.backward()
        
        model.opt.step()

        return loss


def main(name,test=False, chkpt=None):
    memory_size = 1000000
    min_rb_size = 20000
    sample_size = 100
    
    eps_min = 0.01

    eps_decay = 0.999995

    env_steps_before_train = 1000
    tgt_model_update = 100
    



    # if not test:
    #     wandb.init(project="dqn-cartpole",name=name)
    done = False
    

    # done  = False
    if test:
        env = gym.make("CartPole-v1", render_mode= "human")
    else:
        env = gym.make("CartPole-v1")


    last_observation, info = env.reset() 
    import ipdb; ipdb.set_trace()
    m = Model(env.observation_space.shape,env.action_space.n) # model that we train
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))

    # target model to update. Check David Silver lecture
    # for background information
    tgt = Model(env.observation_space.shape,env.action_space.n) # fixed model
    update_tgt_model(m,tgt)

        
    rb = ReplayBuffer()
    steps_since_train = 0
    epochs_since_tgt = 0
    steps_num = -1 * min_rb_size
    # qvals = m(torch.Tensor(last_observation))
    # import ipdb; ipdb.set_trace()


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

            if random() < eps:
                # use completely random action
                action = env.action_space.sample()
            else:
                # use the agent to get the action
                action = m(torch.Tensor(last_observation)).max(-1)[-1].item()

            # action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            rolling_reward += reward

            reward = reward*0.01

            rb.insert(Sarsd(last_observation, action,reward,observation,done))
            last_observation = observation  

            done = terminated or truncated
            if done:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                rolling_reward = 0

                observation, info = env.reset()

            steps_since_train += 1
            steps_num += 1
            
            if (not test) and rb.idx > min_rb_size and steps_since_train > env_steps_before_train:
                
                # it can be seen that the replay is duplicated
                # but this keeps the design of the the replay buffer
                # simple at this point. think of other RBs to improve
                loss=  train_step(m,rb.sample(sample_size),tgt,env.action_space.n)

                # import ipdb; ipdb.set_trace()
                # if not np.isnan(np.mean(episode_rewards)):
                #     wandb.log({'loss':loss.detach().item(), 'eps':eps, 'avg_reward' : np.mean(episode_rewards)},step=steps_num) 
                
                episode_rewards = []
                epochs_since_tgt += 1
                if epochs_since_tgt > tgt_model_update:
                    print("Updating the Target Model")
                    update_tgt_model(m,tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(),f"models/{steps_num}.pth")

                steps_since_train = 0

                

    except KeyboardInterrupt:
        pass

    env.close()
    wandb.finish()




 
if __name__ == '__main__':
    main(False, "models/2348554.pth")
    # argh.dispatch_command(main)