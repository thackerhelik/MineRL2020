import json
import select
import time
import logging
import os
import random


from typing import Callable

import gym
import minerl
import abc
import numpy as np

import coloredlogs
coloredlogs.install(logging.DEBUG)

from model import Model
import torch
import cv2



MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamondVectorObf.

device = "cuda" if torch.cuda.is_available() else "cpu"

# !!! Do not change this! This is part of the submission kit !!!
class EpisodeDone(Exception):
    pass


# !!! Do not change this! This is part of the submission kit !!!
class Episode(gym.Env):
    """A class for a single episode."""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i


class MineRLAgent():
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL

    By default this agent behaves like a random agent: pick random action on
    each step.

    NOTE:
        This class enables the evaluator to run your agent in parallel in Threads,
        which means anything loaded in load_agent will be shared among parallel
        agents. Take care when tracking e.g. hidden state (this should go to run_agent_on_episode).
    """

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        # Some helpful constants from the environment.
        #flat_video_obs_size = 64*64*3
        #obs_size = 64
        #ac_size = 64
        # Load up the behavioural cloning model.
        self.model = Model()
        self.model.load_state_dict(torch.load("train/model.pt", map_location=device))
        # self.model.load_state_dict(torch.load("testing/m.pt", map_location=device))
        
        self.model.to(device)


    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        NOTE:
            This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        
        total_reward = 0
        steps = 0

        with torch.no_grad():
            obs = single_episode_env.reset()
            done = False
            

            state = self.model.get_zero_state(1, device=device)
            s = torch.zeros((1,1,64), dtype=torch.float32, device=device)

            while not done:
                
                spatial = torch.tensor(obs["pov"], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).transpose(2,4)
                nonspatial = torch.cat([torch.tensor(obs["vector"], device=device, dtype=torch.float32), 
                                        torch.ones((2,), device=device,dtype=torch.float32)], dim=0).unsqueeze(0).unsqueeze(0)
                s, state = self.model.sample(spatial, nonspatial, s, state, torch.zeros((1,1,64),dtype=torch.float32,device=device))
                
                for i in range(1):
                    obs,reward,done,_ = single_episode_env.step({"vector":s})
                    total_reward += reward
                    if done:
                        break

                steps += 1
                if steps >= MAX_TEST_EPISODE_LEN:
                    break

        env = single_episode_env

        action_list = np.arange(self.num_actions)
