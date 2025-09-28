#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:00:10 2024

@author: badarinath
"""
 
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import importlib
from itertools import product, combinations
import sys

#%%

sys.path.insert(0, r"C:\Users\innov\OneDrive\Desktop\IDTR-project\Code")

import IVIDTR_curr_modified_V_patched_MIO_debug as VIDTR_module
import markov_decision_processes as mdp_module
import disjoint_box_union
import constraint_conditions
import constraint_conditions as cc
import pandas as pd                                                              
import VIDTR_envs
from VIDTR_envs import GridEnv

                                                    
#%%
                                                                                
importlib.reload(constraint_conditions)
importlib.reload(disjoint_box_union)
importlib.reload(VIDTR_module)
importlib.reload(mdp_module)
importlib.reload(VIDTR_envs)

from markov_decision_processes import MarkovDecisionProcess as MDP
from disjoint_box_union import DisjointBoxUnion as DBU
from IVIDTR_curr_modified_V_patched_MIO_debug import VIDTR

#%%

class VIDTR_grid:
    
    '''
    Build the algorithm environment for the VIDTR on a grid
    
    '''
    
    def __init__(self, dimensions, center, side_lengths, stepsizes, max_lengths,
                 max_complexity, goal, time_horizon, gamma, eta, rho,
                 max_conditions = np.inf, lambda_regs = 0, reward_coeff = 1.0):
        
        '''
        Parameters:
        -----------------------------------------------------------------------
        dimensions : int
                     Dimension of the grid 
        
        center : np.array
                 Center of the grid
                 
        side_lengths : np.array
                       Side lengths of the grid
                       
        stepsizes : np.array
                    Stepsizes for the grid
        
        max_lengths : np.array 
                      Maximum lengths for the grid
        
        max_complexity : int
                         Maximum complexity for the tree 
        
        goal : np.array
               Location of the goal for the 2D grid problem
               
        time_horizon : int
                       Time horizon for the VIDTR problem
                       
        gamma : float
                Discount factor
        
        eta : float
              Splitting promotion constant    
        
        rho : float
              Condition promotion constant

        Stores:
        -----------------------------------------------------------------------
        envs : list[GridEnv]
               The 2D environments for the grid for the different timesteps
        
        VIDTR_MDP : markov_decision_processes
                    The Markov Decision Process represented in the algorithm
        
        algo : VIDTR_algo
               The algorithm representing VIDTR
        '''
        self.dimensions = dimensions
        self.center = center
        self.side_lengths = side_lengths
        self.stepsizes = stepsizes
        self.max_lengths = max_lengths
        self.max_complexity = max_complexity
        self.goal = goal
        self.time_horizon = time_horizon
        self.gamma = gamma
        self.eta = eta
        self.rho = rho
        self.reward_coeff = reward_coeff

        self.env = GridEnv(dimensions, center, side_lengths, goal,
                           stepsizes = stepsizes,reward_coeff=reward_coeff)
        
        self.transitions = [self.env.transition for t in range(time_horizon)]
        self.rewards = [self.env.reward for t in range(time_horizon)]
        
        self.actions = [self.env.actions for t in range(time_horizon)]          
        self.states = [self.env.state_space for t in range(time_horizon)]       
        
        self.VIDTR_MDP = MDP(dimensions, self.states, self.actions, time_horizon, gamma,
                             self.transitions, self.rewards)                    
        
        self.algo = VIDTR(self.VIDTR_MDP, max_lengths, eta, rho, max_complexity,
                          stepsizes, max_conditions = max_conditions,
                          lambda_regs = lambda_regs)
        
        
        print('We initialize the VIDTR algorithm with lambda vals initialized as')
        print(lambda_regs)
        
    
    def generate_random_trajectories(self, N):
        '''
        Generate N trajectories from the VIDTR grid setup where we take a
        random action at each timestep and we choose a random initial state
        
        Returns:
        -----------------------------------------------------------------------
           obs_states : list[list]
                        N trajectories of the states observed
        
           obs_actions : list[list]
                         N trajectories of the actions observed
           
           obs_rewards : list[list]
                         N trajectories of rewards obtained                    
           
        '''
        
        obs_states = []
        obs_actions = []
        obs_rewards = []
        
        for traj_no in range(N):
            obs_states.append([])
            obs_actions.append([])
            obs_rewards.append([])
            s = np.squeeze(self.VIDTR_MDP.state_spaces[0].pick_random_point())  
            obs_states[-1].append(s)
            
            for t in range(self.time_horizon):
                
                a = random.sample(self.actions[t], 1)[0]
                r = self.rewards[t](s,a)
                
                s = self.env.move(s,a)
                obs_states[-1].append(s)
                obs_actions[-1].append(a)
                obs_rewards[-1].append(r)
                
            
        return obs_states, obs_actions, obs_rewards

#%%%%

'''
Tests GridEnv
'''

if __name__ == '__main__':
    
    # Environmental parameters
    
    dimensions = 2
    center = np.array([0, 0])
    side_lengths = np.array([10, 10])
    goal = np.array([0, 0])
    time_horizon = 4
    gamma = 0.9
    max_lengths = [4 for t in range(time_horizon)]
    stepsizes = 1.0
    max_complexity = 2
    dims = [dimensions for i in range(time_horizon)]
    
    
    #%%
    
    # Internal parameters
    
    eta = 0.1
    rho = 0.1
    lambda_regs = [1.5 for t in range(time_horizon+1)]
    max_conditions = np.inf
    reward_coeff = 1.0
    
    grid_class = VIDTR_grid(dimensions, center, side_lengths,
                            stepsizes, max_lengths, max_complexity, goal,
                            time_horizon, gamma, eta, rho, max_conditions = max_conditions,
                            lambda_regs = lambda_regs, reward_coeff=reward_coeff)
    
    #%%                                                                            
    N = 20000
    obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)
        
    #%%
    '''
    Tests VIDTR Bellman function
    '''
    # For t = 3 is the bellman map correct
    s = np.array([2,2])
    max_val = -np.inf
    best_action = grid_class.actions[0]
    
    for a in grid_class.actions[0]:
        bellman_val = grid_class.VIDTR_MDP.reward_functions[2](s,a) 
        if bellman_val > max_val:
            best_action = a
    
    best_action


#%%
    '''
    Tests for compute_optimal_policies
    '''
    points = []
    values = []                                                                  
    
    optimal_policies, optimal_values = grid_class.algo.compute_optimal_policies()
    env = GridEnv(dimensions, center, side_lengths, goal)
    
    print(grid_class.algo.MDP.state_spaces[0])
    
    for t in range(time_horizon):
        print(optimal_policies[t](np.array([0,2])))
    
    
    for t in range(time_horizon):
        env.plot_policy_2D(optimal_policies[t], title = f'Actions at time {t}')

#%%%
    '''
    Tests for storing of optimal_value_funcs
    '''
    
    print(grid_class.algo.optimal_value_funcs)
                                                                                
#%%
    '''
    Tests for printing optimal values and actions
    '''
                                                                                
    for t in range(grid_class.time_horizon):                                    
        states = disjoint_box_union.DBUIterator(grid_class.algo.MDP.state_spaces[t])
        iter_state = iter(states)                                               
    
        print('Optimal actions are')                                             
        for s in iter_state:                                                    
            print(optimal_policies[0](np.array(s)))                                 
    
        print('Optimal values are')
        states = disjoint_box_union.DBUIterator(grid_class.algo.MDP.state_spaces[t])
        iter_state = iter(states)
        for s in iter_state:
            print(f'The value at {s} is {optimal_values[0](np.array(s))}')      
    
    
    #%%
    print('Observed states is')
    print(obs_states[0])

    #%%                                                                        
    '''                                                                        
    Tests for compute_interpretable_policies
    '''
    
    # What is the optimization problem we wish to solve here? 
    
    optimal_conditions, optimal_actions = grid_class.algo.compute_interpretable_policies(obs_states=obs_states,
                                                                                         conditions_string='order_stats',
                                                                                         M = 2 * np.max(side_lengths) + 2,
                                                                                         debug=True) 
                                                                               
      #%%                                                                         
                                                                                
    for t in range(time_horizon):
                                                                                
        print(f'Optimal conditions at {t} is')
        for i, c in enumerate(optimal_conditions[t]):                          
            print(c)                                           
        
        print(f'Optimal actions at {t} is')
        for i, a in enumerate(optimal_actions[t]):                       
            print(a)
                                                                               
    #%%                                                                         
    '''                                                                        
    VIDTR - plot errors                                                        
    '''                                                                        
    grid_class.algo.plot_errors()                                               
                                                                                
    #%%
    '''
    VIDTR - get interpretable policy                                           
    '''
    for t in range(grid_class.time_horizon-1):                                   
        
        int_policy = VIDTR.get_interpretable_policy_conditions(optimal_conditions[t],
                                                               optimal_actions[t])
            
        grid_class.env.plot_policy_2D(int_policy, title=f'Int. policy at time {t}',
                                      saved_fig_name = f'MIO_modified_vidtr_plots_{t}')
        

    #%%
    
    def plot_confidence_intervals(errors_list, title, labels, figure_title):   
        
        num_methods = len(errors_list)                                         
        means = []
        half_std_devs = []

        for method_errors in errors_list:
            method_errors = np.array(method_errors)
            mean = np.mean(method_errors)
            std_dev = np.std(method_errors)                                     
            means.append(mean)                                                  
            half_std_devs.append(std_dev / 2)  # Half of the standard deviation
    
        # Create a plot with error bars (mean ± half std_dev)
        plt.figure(figsize=(8, 5))
        x = np.arange(num_methods)  # X-axis: Method indices
        plt.errorbar(x, means, yerr=half_std_devs, fmt='o', capsize=5)
        
        plt.xticks(x[:len(labels)], labels)  # Custom labels for methods
        plt.xlabel('Integration Method')
        plt.ylabel('Error')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(figure_title)
        plt.show()
    