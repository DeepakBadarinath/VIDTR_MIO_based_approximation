#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:36:08 2024

@author: badarinath
"""

import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import inspect
import math
import sys

sys.path.insert(0, r"C:\Users\innov\OneDrive\Desktop\IDTR-project\Code")

import markov_decision_processes as mdp_module
import disjoint_box_union                                                                            
import constraint_conditions as cc
import box_estimation_mio_problem as BEMP
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D


import itertools
from typing import List, Tuple, Union

from importlib import reload

sys.setrecursionlimit(9000)  # Set a higher limit if needed


#%%

mdp_module = reload(mdp_module)
disjoint_box_union = reload(disjoint_box_union)
cc = reload(cc)
BEMP = reload(BEMP)

from disjoint_box_union import DisjointBoxUnion as DBU

#%%

class VIDTR:
    
    def __init__(self, MDP, max_lengths,
                 etas, rhos, max_complexity,
                 stepsizes, max_conditions = math.inf,
                 lambda_regs = 0):                        
        '''
        Value-based interpretable dynamic treatment regimes; Generate a tree based
        policy while solving a regularized interpretable form of the Bellmann 
        equation with complexities ranging from 1 to max_complexity.
        
        In this module we assume that the state spaces are time dependent.
        
        Parameters:
        -----------------------------------------------------------------------
        MDP : MarkovDecisionProcess
              The underlying MDP from where we want to get interpretable policies
              
        max_lengths : list[T] or int
                      The max depth of the tree upto the T timesteps
        
        etas : list[T] or int
               Volume promotion constants
               Higher this value, greater promotion in the splitting process    
                                                                               
        rhos : list[T] or int                                                            
               Complexity promotion constants                                    
               Higher this value, greater the penalization effect of the complexity 
               splitting process                                                
                                                                               
        max_complexity : int or list                                                   
                         The maximum complexity of the conditions; maximum number of 
                         and conditions present in any condition               
                                                                               
        stepsizes : list[np.array((1, MDP.states.dimension[t])) for t in range(time_horizon)] or float or int        
                    The stepsizes when we have to integrate over the DBU       
        
        max_conditions : int or list                                           
                         The maximum number of conditions per time and lengthstep
                         If None then all the conditions will be looked at     
        
        lambda_regs : list[T]
                      The lambda regularization to use in the obj. function which is
                      Min u.Z + \lambda ||b-a|| at each time step
            
        '''
        
        self.MDP = MDP
        self.time_horizon = self.MDP.time_horizon
        
        if type(max_lengths) == float or type(max_lengths) == int:
            max_lengths = [max_lengths for t in range(self.MDP.time_horizon)]
        
        self.max_lengths = max_lengths
        
        if type(etas) == float or type(etas) == int:
            etas = [etas for t in range(self.MDP.time_horizon)]
        
        self.etas = etas
        
        if type(rhos) == float or type(rhos) == int:
            rhos = [rhos for t in range(self.MDP.time_horizon)]

        self.rhos = rhos
        
        if type(stepsizes) == float or type(stepsizes) == int:
            stepsizes = [np.ones((1, MDP.state_spaces[t].dimension)) for t in range(self.time_horizon)]
        
        self.stepsizes = stepsizes
        
        if type(max_complexity) == int:
            max_complexity = [max_complexity for t in range(self.MDP.time_horizon)]
        
        self.max_complexity = max_complexity
        
        self.true_values = [lambda s: 0 for t in range(self.MDP.time_horizon+1)]
        
        if ((type(max_conditions) == int) or (max_conditions == math.inf)):
            max_conditions = [max_conditions for t in range(self.MDP.time_horizon)]
        
        self.max_conditions = max_conditions
        
        if ((lambda_regs == 0) or (type(lambda_regs) == int) or (type(lambda_regs) == float)):
            self.lambda_regs = [lambda_regs for t in range(self.MDP.time_horizon+1)]
        
        self.lambda_regs = lambda_regs
        
        #print(f'The max conditions is {self.max_conditions}')
        
    def maximum_over_actions(self, function, t):
        
        '''
        Given a function over states and actions, find the function only over
        states by maximizing over the actions.
        
        Parameters:
        -----------------------------------------------------------------------
        function : function(s,a)
                   A function over states and actions for which we wish to get 
                   the map s \to max_A f(s,a)

        Returns:
        -----------------------------------------------------------------------
        max_function : function(s)
                       s \to \max_A f(s,a) is the function we wish to get
        
        '''
        def max_function(s):
            
            max_val = -np.inf
            
            for a in self.MDP.action_spaces[t]:
                val = function(np.array(s), a)
                if val > max_val:
                    max_val = val
            
            return max_val
                    
        return max_function


    def bellman_equation(self, t):
        '''
        Return the Bellman equation for the Markov Decision Process.   
        
        Assumes we know the true values from t+1 to T.                         
        
        Parameters:                                                                
        -----------------------------------------------------------------------
        t : float                                                               
            The time at which we wish to return the Bellman function for the MDP.
                                                                               
        Returns:                                                               
        -----------------------------------------------------------------------
        bellman_function : func                                                
                           The Bellman function of the MDP for the t'th timestep.

        '''
        def bellman_map(s,a):                                                   
            
            curr_space = self.MDP.state_spaces[t]
            
            if len(self.MDP.state_spaces) <= (t+1):
                new_space = curr_space
            else:
                new_space = self.MDP.state_spaces[t+1]
                                   
            action_space = self.MDP.action_spaces[t]                           
            
            dbu_iter_class = disjoint_box_union.DBUIterator(new_space)              
            dbu_iterator = iter(dbu_iter_class)                                
            
            if (t == (self.MDP.time_horizon - 1)):
            
                return self.MDP.reward_functions[t](np.array(s), a, curr_space, action_space)
            
            else:
                
                r = self.MDP.reward_functions[t](np.array(s), a, curr_space, action_space)
                T_times_V = 0
                for s_new in dbu_iterator:
                
                    kernel_eval = self.MDP.transition_kernels[t](np.array(s_new), np.array(s), a, curr_space, action_space)
                    vals_element = self.optimal_value_funcs[t+1](np.array(s_new))
                    
                    adding_element = kernel_eval * vals_element
                    T_times_V += adding_element
                
                
                return r + self.MDP.gamma * T_times_V             
        
        return bellman_map                                                     

    
    def bellman_equation_I(self, t):
        
        '''
        Return the interpretable Bellman equation for the Markov Decision Process.
        
        Assumes we know the interpretable value function from timestep t+1 to T.
        
        Parameters:
        -----------------------------------------------------------------------
        t : float
            The time at which we wish to return the interpretable Bellman equation
        
        Returns:
        -----------------------------------------------------------------------
        int_bellman_function : func
                               The Interpretable Bellman function for the MDP for the t'th timestep
        
        '''
                
        #print(f'We ask to evaluate the bellman map at timestep {t}')
        
        def bellman_map_interpretable(s,a):                                                   
            
            curr_space = self.MDP.state_spaces[t]                                   
            
            if (len(self.MDP.state_spaces) <= (t+1)):
                new_space = curr_space
            else:
                new_space = self.MDP.state_spaces[t+1]
            
            action_space = self.MDP.action_spaces[t]                           
            
            dbu_iter_class = disjoint_box_union.DBUIterator(new_space)              
            dbu_iterator = iter(dbu_iter_class)                                
            
            if t == self.MDP.time_horizon-1:
                
                return self.MDP.reward_functions[t](np.array(s), a, curr_space, action_space)
            
            else:
                                                                                
                return self.MDP.reward_functions[t](np.array(s), a, curr_space, action_space) + self.MDP.gamma * (
                            np.sum([self.MDP.transition_kernels[t](np.array(s_new), np.array(s), a, curr_space, action_space) * self.int_value_functions[t+1](np.array(s_new))
                            for s_new in dbu_iterator]))                        
        
        return bellman_map_interpretable                                        

    @staticmethod                                                              
    def fix_a(f, a):
        '''
        Given a function f(s,a), get the function over S by fixing the action   
                                                                                                                                                         
        Parameters:                                                                                                                                               
        -----------------------------------------------------------------------
        f : func                                                               
            The function we wish to get the projection over                    
            
        a : type(self.MDP.actions[0])                                          
            The action that is fixed                                           
        '''
        return lambda s : f(s,a)                                               
    
    
    @staticmethod
    def redefine_function(f, s, a):                                            
        '''
        Given a function f, redefine it such that f(s) is now a                
                                                                                
        Parameters:                                                            
        -----------------------------------------------------------------------
        f : function                                                           
            Old function we wish to redefine                                   
        s : type(domain(function))                                             
            The point at which we wish to redefine f                           
        a : type(range(function))                                                
            The value taken by f at s                                          
 
        Returns:                                                                  
        -----------------------------------------------------------------------
        g : function                                                           
            Redefined function                                                 
                                                                                
        '''
        def g(state):
            if np.sum((np.array(state)-np.array(s))**2) == 0:
                return a
            else:
                return f(state)
        return g
        
    @staticmethod
    def convert_function_to_dict_s_a(f, S):
        '''
        Given a function f : S \times A \to \mathbb{R}                         
        Redefine it such that f is now represented by a dictonary              

        Parameters:                                                            
        -----------------------------------------------------------------------
        f : function                                                           
            The function that is to be redefined to give a dictonary           
            
        S : iterable version of the state space                                
            iter(DisjointBoxUnionIterator)                                     

        Returns:
        -----------------------------------------------------------------------
        f_dict : dictionary
                The function which is now redefined to be a dictonary

        '''
        f_dict = {}
        for s in S:
            f_dict[tuple(s)] = f(s)
        
        return f_dict
    
    @staticmethod
    def convert_dict_to_function(f_dict, S, default_value=0):
        '''
            
        Given a dictonary f_dict, redefine it such that we get a function f from 
        S to A

        Parameters:
        -----------------------------------------------------------------------
        f_dict : dictionary
                 The dictonary form of the function
                 
        S : iterable version of the state space
            iter(DisjointBoxUnionIterator)

        Returns:
        -----------------------------------------------------------------------
        f : func
            The function version of the dictonary

        '''
            
        def f(s):
            
            if tuple(s) in f_dict.keys():
                
                return f_dict[tuple(s)]
            
            else:
                return default_value
        
        return f
    
    @staticmethod
    def process_state(s, action0):
        return tuple(s), (0, action0)

    
    def memoizer(self, t, f):
        
        '''
        Given a function f over the MDP state space at time t, create it's memoized version
        We do this by creating f_dict where we have f_dict[tuple(s)] = f(s)
        
        '''
        
        dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
        
        state_iterator = iter(dbu_iter_class)
        
        f_dict = {}
        
        for s in state_iterator:
            
            f_dict[tuple(s)] = f(s)
        
        f_memoized = VIDTR.convert_dict_to_function(f_dict, state_iterator)
        return f_memoized
    
    def compute_optimal_policies(self):
        '''
        Compute the true value functions at the different timesteps.
        
        Stores:
        -----------------------------------------------------------------------
        optimal_values : list[function]
                         A list of length self.MDP.time_horizon which represents the 
                         true value functions at the different timesteps
        
        optimal_policies : list[function]
                           The list of optimal policies for the different timesteps 
                           for the MDP
        '''
        #zero_value = lambda s : 0
        zero_value_dicts = []
        const_action_dicts = []
        
        zero_func = lambda s : 0
        
        self.optimal_policy_funcs = [zero_func for t in range(self.MDP.time_horizon)]
        self.optimal_value_funcs = [zero_func for t in range(self.MDP.time_horizon)]
        
        
        for t in range(self.time_horizon):
            
            print(f't is {t} and the state space is')
            print(self.MDP.state_spaces[t])
            
            #Setting up this iter_class is taking a lot of time-why?
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)
            zero_dict = {}
            const_action_dict = {}
            for s in state_iterator: 
                zero_dict[tuple(s)] = 0 
                const_action_dict[tuple(s)] = self.MDP.action_spaces[t][0] #Parallelize the loop over state_iterator
                
            zero_value_dicts.append(zero_dict)
            const_action_dicts.append(const_action_dict)
        
        optimal_policy_dicts = const_action_dicts
        
        optimal_value_dicts = zero_value_dicts
        
                                                                         
        for t in np.arange(self.time_horizon-1, -1, -1):     
            
            print(f'Value iteration at time {t}')
            
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            
            state_iterator = iter(dbu_iter_class)
            
            for s in state_iterator:
                print(f'We are at state {s}')
                max_val = -np.inf
                
                for a in self.MDP.action_spaces[t]:
                                                    
                    bellman_value = self.bellman_equation(t)(s,a)
                    
                    if bellman_value > max_val:                                 
                        
                        max_val = bellman_value
                        optimal_action = a
                                                                                
                        optimal_policy_dicts[t][tuple(s)] = optimal_action                       
                        optimal_value_dicts[t][tuple(s)] = max_val                        
            
            print('State count iteration done')
            dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator = iter(dbu_iter_class)
            
            optimal_policy_func = VIDTR.convert_dict_to_function(optimal_policy_dicts[t],
                                                                 state_iterator)
            
            dbu_iter_class_1 = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
            state_iterator_new = iter(dbu_iter_class_1)
            
            optimal_value_func = VIDTR.convert_dict_to_function(optimal_value_dicts[t],
                                                                state_iterator_new)
            
            self.optimal_policy_funcs[t] = optimal_policy_func
            self.optimal_value_funcs[t] = optimal_value_func
        
        return self.optimal_policy_funcs, self.optimal_value_funcs
    
    def constant_eta_function(self, t):                                         
        '''
        Return the constant \eta function for time t

        Parameters:
        -----------------------------------------------------------------------
        t : int                                                                
            Time step                                                           

        Returns:
        -----------------------------------------------------------------------
        f : function
            Constant eta function at time t                                    
                                                                                 
        '''                                                                                               
        f = lambda s,a: self.etas[t]
        return f
    
    @staticmethod
    def fixed_reward_function(t, s, a, MDP, debug = False):
        
        '''
        For the MDP as in the function parameter, return the reward function.
        
        Parameters:
        -----------------------------------------------------------------------
        t : int
            Timestep at which we return the reward function
        
        s : state_point
            The point on the state space at which we return the reward function
        
        a : action
            The action we take at this reward function
        
        MDP : MarkovDecisionProcess
              The Markov Decision Process for which we compute the fixed reward function  
        
        '''
        
        if debug:
            
            print(f'For timestep {t}, state {s}, and action {a} the reward')
            print(MDP.reward_functions[t](s, a, MDP.state_spaces[t], MDP.action_spaces[t]))
        
        return MDP.reward_functions[t](s, a, MDP.state_spaces[t],
                                       MDP.action_spaces[t])
    
    @staticmethod
    def bellman_value_function_I(t, s, a, MDP, int_policy,
                                 int_value_function_next, debug = False):
        
        '''
        For the MDP, return the interpretable bellman_value_function at timestep t.
        
        Parameters:
        -----------------------------------------------------------------------
        t : int
            The timestep at which we compute the Bellman value function.
        
        s : state_point
            The point on the state space at which we return the Bellman value function.
        
        a : action
            The action we take on the Bellman value function.
        
        MDP : MarkovDecisionProcess
              The Markov Decision Process for which we compute the Bellman value function.
        
 int_policy : func
              The interpretable policy at the next timestep.
              
        '''
        
        space = MDP.state_spaces[t]                                   
        action_space = MDP.action_spaces[t]                           
        
        if len(MDP.state_spaces) <= t+1:
            iter_space = MDP.state_spaces[t]
        else:
            iter_space = MDP.state_spaces[t+1]
                                                                                
        
        dbu_iter_class = disjoint_box_union.DBUIterator(iter_space)              
        dbu_iterator = iter(dbu_iter_class)                                     
        
        if debug:

            total_sum = MDP.reward_functions[t](np.array(s), a, space, action_space)
            for s_new in dbu_iterator:                                         
                
                total_sum += int_value_function_next(np.array(s_new)) * MDP.transition_kernels[t](np.array(s_new), np.array(s), a, space, action_space)
            
            
        return MDP.reward_functions[t](np.array(s), a, space, action_space) + MDP.gamma * (
                np.sum([[MDP.transition_kernels[t](np.array(s_new), np.array(s), a, space, action_space) * int_value_function_next(np.array(s_new)) 
                        for s_new in dbu_iterator]]))
        
    
    @staticmethod
    def last_step_int_value_function(t, int_policy, MDP, debug = False):
        
        def last_step_val_function(s):
            
            return VIDTR.fixed_reward_function(t, s, int_policy(s),
                                               MDP, debug=debug)
        
        return last_step_val_function
    
    
    @staticmethod
    def general_int_value_function(t, int_policy, MDP,
                                   next_int_val_function, debug = False):
        
        def interpretable_value_function(s):
            return VIDTR.bellman_value_function_I(t, s, int_policy(s),
                                                  MDP, int_policy,
                                                  next_int_val_function,
                                                  debug=debug)
        
        return interpretable_value_function

    
    def int_value_checks(self, t, int_value_function, int_policy):
        
        '''
        Check whether the int_value_function for time t and for int_policy is 
        greater than the optimal_value_function for the same timestep.
        '''
    
        dbu_iter_class = disjoint_box_union.DBUIterator(self.MDP.state_spaces[t])
        #ic('DBU iter class computation done')
        state_iterator = iter(dbu_iter_class)
        
        for s in state_iterator:
            
            #ic(f'Int. value function and optimal_value_function at state {s} and time {t}')
            int_value = int_value_function(s)
            optimal_value = self.optimal_value_funcs[t](s)
            
            #ic(int_value, optimal_value)
            
            if int_value > optimal_value:
                
                #ic(f'For state {s} and time {t}, we have that the int value function {int_value} > {optimal_value}')
                optimal_value = self.optimal_value_funcs[t](s)
                raise ValueError(f"Int_value_function at point {s} = {int_value} is greater than optimal_value_function at time {t} and state {s}")


    def derive_condition(self, sigma_vals, tau_vals, non_zero_indices, dimension):
        
        '''
        Given sigma_vals and tau_vals of dimension q, check if tau[i] >= sigma[i]. if not reverse it.
        Let non_zero_indices be a vector of dimension q such that which represents the dimensions under question.
        
        Further derive the constraint condition associated with these indices, sigma and tau values.

        Parameters:
-------------------------------------------------------------------------------
        sigma_vals : list[np.array[d_1], ..., np.array[d_q]]
                     Lower bounds for the condition
    
        tau_vals : np.array[np.array[d_1], ..., np.array[d_q]]
                   Upper bounds for the condition

              q : int
                  The number of parameters considered in the state space.
                  
        Returns:
-------------------------------------------------------------------------------
        Condition : ConstraintCondition
                    I [ sigma_0 < \vec{X}_{i0} \leq tau_0, ..., sigma_q < \vec{X}_{iq} \leq tau_{iq} ]             
        
        '''
        
        
        for i in range(len(sigma_vals)):
            
            if sigma_vals[i] > tau_vals[i]:
                
                x = sigma_vals[i]
                sigma_vals[i] = tau_vals[i]
                tau_vals[i] = x
        
        bounds = np.stack((np.array(sigma_vals),
                           np.array(tau_vals)), axis=1)  # Shape: (q, 2)
        
        
        condition = cc.ConstraintConditions(dimension = dimension,
                                            non_zero_indices = non_zero_indices,
                                            bounds = bounds)
        return condition
        
    @classmethod
    def generate_tuples(self, d, q):
        domain = range(d)
        
        
        all_tuples = itertools.chain.from_iterable(
            (itertools.product(domain, repeat=k) for k in range(1, q + 1))
        )
        return all_tuples
    
    
    def compute_U_vals(self, action, t, obs_states, remaining_space):
        
        '''
        Compute U values for the different times when we know the action,
        trajectories, and the remaining space
        
        Parameters:
        -----------------------------------------------------------------------
        action - Element of action_spaces[t]
                 The action for which we wish to compute the U_vals
        
        t - timestep
            The timestep at which we compute the U_vals
        
        obs_states - list[list]
                     The observed states at the different time and lengthsteps.
                     obs_states[i] := i'th trajectory and we have that obs_states[i][t] is the t'th point in this trajectory 
                     This assumes a 1D state space at each t'th timestep
        
        remaining_space - DisjointBoxUnion
                          The remaining space at that time and length instance
        
        '''
        
        neg_eta_function = lambda s :  -self.etas[t]
        maxim_bellman_function = lambda s: self.maximum_over_actions(self.bellman_equation(t), t)(s)
        fixed_bellman_function = lambda s: -VIDTR.fix_a(self.bellman_equation_I(t), a=action)(s)
        total_bellman_function = lambda s: maxim_bellman_function(s) + fixed_bellman_function(s)
        integ_function = lambda s: total_bellman_function(s) + neg_eta_function(s)
        
        U_vals = []
        points_at_t = []
        
        for traj_no, traj in enumerate(obs_states):
            
            point = traj[t]
            points_at_t.append(point)
            
            if remaining_space.is_point_in_DBU(point):    
                U_val = float(integ_function(point))
            else:
                U_val = 0
            
            U_vals.append(U_val)
        
        return U_vals
    
    
    def visualize_selection(self, X, U, selected, a, b, title = ''):
        
        '''
        Visualize the selected U_values
        '''
        
        print('We wish to visualize')
        print(X)
        
        print('For U_vals')
        print(U)
        
        print('Selected')
        print(selected)
        
        print('Left vals')
        print(a)
        
        print('Right vals')
        print(b)
        
        if (type(X) == list):
            X = np.array(X)
            
        print(f'Type and shape of X is {type(X)} and {X.shape}')
        q = X.shape[1]
        
        if len(X.shape) > 2:
            X = np.reshape(X, (X.shape[0], X.shape[1]))
            print('New shape of X is')
            print(X.shape)
            
        if q == 2:
            self._plot_2d(X, U, selected, a, b, title = title)
            
        elif q == 3:
            self._plot_3d(X, U, selected, a, b, title = title)

    def _plot_2d(self, X, U, selected, a, b, title):
        
        '''
        Plot the U_values in 2D.
        '''
        
        plt.figure(figsize=(8, 6))
        colors = ['green' if u < 0 else 'red' for u in U]
        plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k', s=80)
        for i, (x, y) in enumerate(X):
            plt.text(x + 0.1, y, f"{round(U[i], 2):g}", fontsize=8)
        if selected:
            plt.scatter(X[selected, 0], X[selected, 1], color='blue', s=100, label='Selected')
    
        rect = patches.Rectangle((a[0], a[1]), b[0] - a[0], b[1] - a[1],
                                 linewidth=2, edgecolor='black', facecolor='none', label='Box')
        plt.gca().add_patch(rect)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def _plot_3d(self, X, U, selected, a, b, title):
        
        '''
        Plot the U_values in 3D.
        
        '''
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['green' if u < 0 else 'red' for u in U]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=60, edgecolors='k')
        if selected:
            ax.scatter(X[selected, 0], X[selected, 1], X[selected, 2], c='blue', s=100, label='Selected')
    
        # draw bounding box
        from itertools import product
        corners = np.array(list(product(*zip(a, b))))
        for i in range(8):
            for j in range(i+1, 8):
                d = np.abs(corners[i] - corners[j])
                if np.count_nonzero(d) == 1:
                    ax.plot(*zip(corners[i], corners[j]), color="black", linewidth=1)
    
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("x₃")
        ax.set_title(title)
        plt.legend()
        plt.show()
    
    
    def compute_interpretable_policies(self, 
                                       debug = False,
                                       obs_states=None,
                                       conditions_string='order_stats',
                                       M = 100):                                                               
        
        '''                                                                    
        Compute the interpretable policies for the different length and        
        timesteps given a DBUMarkovDecisionProcess.                             
        
        Parameters:                                                            
        -----------------------------------------------------------------------
        cond_stepsizes : int or float or np.array(self.DBU.dimension)               
                         The stepsizes over the conditional DBU                     
                                                                                 
        integration_method : string
                             The method of integration we wish to use -
                             trajectory integratal versus DBU based integral
        
        integral_percent : float
                           The percent of points we wish to sample from
        
        debug : bool
                Add additional print statements to debug the plots
        
        obs_states : list[list]
                     The observed states at the different time and lengthsteps.
                     obs_states[i] := i'th trajectory and we have that obs_states[i][t] is the t'th point in this trajectory 
                     This assumes a 1D state space at each t'th timestep
        
        conditions_string : string
                            'all' or 'order_stats' -> Use all possible conditions by going over
                            each dimension and state_differences versus conditions given by those over
                            the order statistics
        
        M : int
            The constant to be used in the BEMP
        
        Stores:
        -----------------------------------------------------------------------
        optimal_conditions :  list[list]
                              condition_space[t][l] gives the optimal condition at
                              timestep t and lengthstep l
        
        optimal_errors : list[list]
                         errors[t][l] represents the error obtained at timestep t and 
                         length step l
        
        optimal_actions : list[list]
                          optimal_intepretable_policies[t][l] denotes
                          the optimal interpretable policy obtained at 
                          time t and length l
        
        stored_DBUs :   list[list]
                        stored_DBUs[t][l] is the DBU stored at timestep t and  
                        lengthstep l for the final policy
        
        stepsizes :  np.array(self.DBU.dimension) or int or float
                     The length of the stepsizes in the different dimensions
        
        total_error : list[list]
                      total_error[t][l] := Error incurred at timestep t and lengthstep l. 

        int_policies : list[list[function]]
                       optimal_intepretable_policies[t][l] denotes
                       the optimal interpretable policy obtained at 
                       time t and length l
        
        Returns:
        -----------------------------------------------------------------------
        optimal_conditions, optimal_actions : Optimal condition spaces and optimals
                                              for the given time and lengthstep
                                              respectively
        
        '''
        
        stored_DBUs = []
        optimal_conditions = []
        optimal_actions = []
        
        int_policies = []
        zero_func = lambda s : 0
        int_value_functions = [zero_func for t in range(self.MDP.time_horizon)]
        
        optimal_errors = []
        optimal_fixed_bellman_errors = []
        optimal_maxim_bellman_errors = []
        
        print(f'M is {M}')
        print(f'Lambda_regs is {self.lambda_regs}')
        
        optimal_sigmas = pd.DataFrame(columns=['timestep', 'lengthstep',
                                               'action', 'sigma_vals', 'tau_vals','obj_val'])
        
        remaining_spaces = pd.DataFrame(columns=['timestep', 'lengthstep',
                                                 'no_of_boxes', 'lengths', 'centres',
                                                 'og_dbu', 'subtract_dbu'])
        
        for t in np.arange(self.MDP.time_horizon-1, -1, -1):
            
            print(f'Time is {t}')
            
            all_conditions = []
            all_condition_DBUs = []
            int_policies = [[] ,*int_policies]
            
            
            all_condition_DBUs = []
            
            state_bounds = self.MDP.state_spaces[t].get_total_bounds()
            #ic(state_bounds)
                        
            total_error = 0.0
            
            all_condition_dicts = {}
            
            maxim_bellman_error_dict = {}
            fixed_bellman_error_dict = {}
            
            for i,c in enumerate(all_conditions):
                if c != None:
                    con_DBU = DBU.condition_to_DBU(c, self.stepsizes[t])
                    if con_DBU.no_of_boxes != 0:
                        all_condition_DBUs.append(con_DBU)
                        necc_tuple = con_DBU.dbu_to_tuple()
                        if necc_tuple not in all_condition_dicts:
                            all_condition_dicts[necc_tuple] = 1
            
                                                                                
            optimal_errors = [[], *optimal_errors]
            stored_DBUs = [[], *stored_DBUs]
            optimal_conditions = [[], *optimal_conditions]
            optimal_actions = [[], *optimal_actions]
            
            condition_DBUs = all_condition_DBUs
            conditions = all_conditions
            optimal_cond_DBU = None
            remaining_space = self.MDP.state_spaces[t]

            for l in range(self.max_lengths[t]-1):
                
                min_error = np.inf
                optimal_condition = None
                optimal_action = None
                no_of_null_DBUs = 0
                
                print(f'We are at timestep {t} and lengthstep {l}')
            
                # Loop over all q \leq max_complexity
                
                # Loop over actions
                
                # Loop over trajectories
                
                # Create vector U_ia(X), remember that the optim problem we have is \sum U_{ia} I[X_{it} \in R - G_{tl}], in here we pull
                # out I[x \in G_{tl}] to the U_{ia}. Compute U_{ia} accordingly, some U_{ia} can be zero if X_{it} is not in the right place
                
                # U_{iat} := max{\alpha} [r_t(X_{it}, \alpha) +\gamma P^{\alpha}_t V_{t+1}(X_{it})] - [r_t(X_{it}, A_{it}) + \gamma P^{A_{it}}_t V^I_{t+1}(X_{it})] - \eta
                
                # Finish trajectory loop
                
                # Solve optimization problem to find the optimal condition for corr. lengthstep:
                # Looks like \sum_{i=1}^N U_{ia} I[ sigma_1 < X_{i1} < tau_1, ..., sigma_q < X_{iq} < tau_q ]
                
                # Add complexity constraint
                
                # Update G_{tl} which we later use in the computations as G'_{tl} := G_{tl} \cup \hat{R}_{tl}
                
                neg_eta_function = lambda s :  -self.etas[t]
                optimal_condition = None
                best_non_zero_indices = []
                
                # Loop over all complexities less than equal to max_complexity
                dimension = self.MDP.state_spaces[t].dimension
                optimal_action = None
                optimal_c_tuple = None
                
                points_at_t = []
                
                for traj_no, traj in enumerate(obs_states):
                    
                    point = traj[t]
                    points_at_t.append(point)
                
                for act_no, a in enumerate(self.MDP.action_spaces[t]):
                    
                    U_vals = self.compute_U_vals(a, t, obs_states, remaining_space)
                    
                    print(f'For action {a}, time {t}, we have U_vals to be')
                    print(U_vals)
                    
                    if ((t == self.MDP.time_horizon - 1) and np.array_equal(a, np.array([1, 0])) and (l == 0)):
                       
                       np.save("Stored_U_vals.npy", U_vals)
                       np.save("Points_at_3.npy", np.array(points_at_t))
                       np.save("Upper_bound_M.npy", M)
                       np.save("Lambda_regs.npy", self.lambda_regs[t])

                    
                    indices, sigma_vals, tau_vals, obj_val = BEMP.solve_mip_with_box(U_vals, np.array(points_at_t),
                                                                                     M = M,
                                                                                     lambda_reg=self.lambda_regs[t])
                    
                    self.visualize_selection(np.array(points_at_t), U_vals, indices, sigma_vals,
                                             tau_vals, title = f'Visualize_action_{a}_Time_{t}_Length_{l}')
                    
                    complexity = 0
                    c_tuple = []
                    for i in range(len(sigma_vals)):
                        if (sigma_vals[i] < tau_vals[i]):
                            complexity = complexity + 1
                            c_tuple.append(i)
                    
                    total_cost = obj_val + (self.rhos[t] * complexity * len(np.array(points_at_t)))
                    
                    # Define new row values
                    new_row = {
                        'timestep': t,
                        'lengthstep': l,
                        'action': a,
                        'sigma_vals': sigma_vals,
                        'tau_vals': tau_vals,
                        'obj_val': total_cost
                    }
                    
                    # Append new row to DataFrame
                    optimal_sigmas = pd.concat([optimal_sigmas, pd.DataFrame([new_row])], ignore_index=True)
                                       
                    if total_cost < min_error:
                        min_error = total_cost
                        optimal_action = a
                        optimal_c_tuple = c_tuple
                  
                print(f'For length {l} and time {t} we have optimal action = {optimal_action}')
                U_vals = self.compute_U_vals(optimal_action, t, obs_states, remaining_space)
                
                indices, sigma_vals, tau_vals, obj_val = BEMP.solve_mip_with_box(U_vals, np.array(points_at_t),
                                                                                 M = M, lambda_regs = self.lambda_regs[t])

                complexity = 0
                c_tuple = []
                for i in range(len(sigma_vals)):
                    if (sigma_vals[i] < tau_vals[i]):
                        complexity = complexity + 1
                        c_tuple.append(i)
                
                total_cost = obj_val + (self.rhos[t] * complexity)
                
                print('Sigma vals is')
                print(sigma_vals)
                
                print('Tau vals is')
                print(tau_vals)
                
                if total_cost < min_error:
                    min_error = total_cost
                    optimal_action = a
                    optimal_c_tuple = c_tuple
                    
                optimal_condition = cc.ConstraintConditions(dimension, non_zero_indices = np.array(range(dimension)),
                                                            bounds = (np.array([sigma_vals, tau_vals])).T)
                
                optimal_cond_DBU = disjoint_box_union.DisjointBoxUnion.condition_to_DBU(optimal_condition,
                                                                                        stepsizes=self.stepsizes[t])
                
                print(f'We subtract from {remaining_space} the DBU {optimal_cond_DBU}')
                og_dbu = remaining_space
                remaining_space = remaining_space.subtract_DBUs(optimal_cond_DBU)
                
                space_row = {'timestep': t,
                             'lengthstep': l,
                             'no_of_boxes': remaining_space.no_of_boxes,
                             'lengths': remaining_space.lengths,
                             'centres': remaining_space.centres,
                             'og_dbu': og_dbu,
                             'subtract_dbu': optimal_cond_DBU}
                
                total_error += min_error
                
                # Append new row to DataFrame
                remaining_spaces = pd.concat([remaining_spaces, pd.DataFrame([space_row])], ignore_index=True)
                
                print(f'Timestep {t} and lengthstep {l}:')                      
                print('----------------------------------------------------------------')
                print(f'Optimal condition at timestep {t} and lengthstep {l} is {optimal_condition}')
                print(f'Optimal action at timestep {t} and lengthstep {l} is {optimal_action}')
                print(f'Optimal conditional DBU at timestep {t} and lengthstep {l} is {optimal_cond_DBU}')
                print(f'Optimal error is {min_error}')                          
                print(f'Non null DBUs = {len(condition_DBUs)} - {no_of_null_DBUs}')
                print(f'Eta is {self.etas[t]}, Rho is {self.rhos[t]}')

                
                if len(optimal_errors) == 0:
                    optimal_errors = [[min_error]]
                else:
                    optimal_errors[0].append(min_error)                         
                
                if len(stored_DBUs) == 0:
                    stored_DBUs = [[optimal_cond_DBU]]
                else:
                    stored_DBUs[0].append(optimal_cond_DBU)

                if len(optimal_conditions) == 0:
                    optimal_conditions = [[optimal_condition]]
                else:
                    optimal_conditions[0].append(optimal_condition)                
            
                if len(optimal_actions) == 0:
                    optimal_actions = [[optimal_action]]
                else:
                    optimal_actions[0].append(optimal_action)    

                # Include what happens when remaining space is NULL

                if (remaining_space.no_of_boxes == 0):                                    
                    print('--------------------------------------------------------------')
                    print(f'For timestep {t} we end at lengthstep {l}')

                    int_policy = VIDTR.get_interpretable_policy_conditions(optimal_conditions[0],
                                                                           optimal_actions[0])
                    
                    int_policies = [int_policy] + int_policies
                    
                    
                    if t == self.MDP.time_horizon - 1:
                        
                        int_value_function = VIDTR.last_step_int_value_function(t, int_policy,
                                                                                self.MDP, debug = debug)
                        
                        int_value_function = self.memoizer(t, int_value_function)
                        
                        int_value_functions = [int_value_function] + int_value_functions
                        self.int_value_functions = int_value_functions
                        
                        self.int_value_checks(t, int_value_function, int_policy)
                    
                    else:
                        
                        int_value_function = VIDTR.general_int_value_function(t, int_policy,
                                                                              self.MDP, int_value_functions[t+1], debug = debug)
                        
                        int_value_function = self.memoizer(t, int_value_function)
                        
                        int_value_functions = [int_value_function] + int_value_functions
                        self.int_value_functions = int_value_functions
                        
                        self.int_value_checks(t, int_value_function, int_policy)
                    
                    break
                              
            #Final lengthstep - We can only choose the optimal action here and we work over S - \cap_{i=1}^K S_i
            
            # What happens here is that the remaining region is now just S - G_{tl}
            # We minimize over all the possible actions that can be chosen here.
            if remaining_space.no_of_boxes != 0:
                min_error = np.inf
                best_action = None
                
                for act_no, a in enumerate(self.MDP.action_spaces[t]):
                    
                    maxim_bellman_function = lambda s: self.maximum_over_actions(self.bellman_equation(t), t)(s)
                    fixed_bellman_function = lambda s: -VIDTR.fix_a(self.bellman_equation_I(t), a=a)(s)
                    total_bellman_function = lambda s: maxim_bellman_function(s) + fixed_bellman_function(s)
                    integ_function = lambda s: total_bellman_function(s) + neg_eta_function(s)
                    total_cost = remaining_space.integrate(integ_function)
                    
                    if total_cost < min_error:
                        min_error = total_cost
                        optimal_action = a
                
                total_error += min_error
            
            print('--------------------------------------------------------')
            print(f'Optimal action at timestep {t} and lengthstep {l} is {optimal_action}')
            print(f'Total Optimal Error is {total_cost}')
            
            optimal_errors[0].append(min_error)
            stored_DBUs[0].append(optimal_cond_DBU)
            optimal_conditions[0].append(optimal_condition)
            optimal_actions[0].append(optimal_action)
                   
            int_policy = VIDTR.get_interpretable_policy_conditions(optimal_conditions[0],
                                                                   optimal_actions[0])
            
            int_policies = [int_policy] + int_policies
            
            print('We pre-append the following')
            print(int_policy)
            
            print(f'We are at the timestep {t} and the time horizon is {self.MDP.time_horizon}')
            print(f'We check {t == (self.MDP.time_horizon - 1)}')
            
            if (t == self.MDP.time_horizon - 1):
                
                int_value_function = VIDTR.last_step_int_value_function(t, int_policy, self.MDP, debug=debug)
                
                int_value_function = self.memoizer(t, int_value_function)
                
                int_value_functions = [int_value_function] + int_value_functions
                self.int_value_functions = int_value_functions
                
                
                self.int_value_checks(t, int_value_function, int_policy)
                
            else:
                
                print(f'We reach the else clause for checking {t == (self.MDP.time_horizon - 1)}')
                
                int_value_function = VIDTR.general_int_value_function(t, int_policy,
                                                                      self.MDP, int_value_functions[t+1], debug=debug)
                
                int_value_function = self.memoizer(t, int_value_function)
                
                int_value_functions = [int_value_function] + int_value_functions
                self.int_value_functions = int_value_functions
                
                self.int_value_checks(t, int_value_function, int_policy)
        
        print('We save optimal sigmas next')
        optimal_sigmas.to_csv("optimal_sigma_tau.csv", index=False)

        print('We save remaining spaces next')
        remaining_spaces.to_csv('remaining_spaces.csv', index=False)
        
        self.optimal_conditions = optimal_conditions
        self.optimal_errors = optimal_errors
        self.optimal_actions = optimal_actions
        self.stored_DBUs = stored_DBUs
        self.total_error = total_error
        self.int_policies = int_policies
        
        return optimal_conditions, optimal_actions
    
    @staticmethod
    def get_interpretable_policy_conditions(conditions, actions):
        '''                                                                    
        Given the conditions defining the policy, obtain the interpretable policy
        implied by the conditions.                                             
        
        Parameters:
        -----------------------------------------------------------------------
        conditions : np.array[l]
                     The conditions we want to represent in the int. policy             
        
        actions : np.array[l]
                  The actions represented in the int. policy
                                                                               
        '''

        def policy(state):
            
            for i, cond in enumerate(conditions):                               
                                                                                 
                if cond.contains_point(state):                                  
                                                                                
                    return actions[i]                                           
            
                                                                                
            return actions[-1]                                                  
        
        return policy
    
    @staticmethod
    def get_interpretable_policy_dbus(stored_dbus, actions):
        '''
        Given the dbus defining the policy, obtain the interpretable policy implied
        by the dbus.
        
        Parameters:
        -----------------------------------------------------------------------
        stored_dbus : list[DisjointBoxUnion]
                      The dbus we wish to derive an int. policy out of
        
        actions : list[np.array]
                  The actions we wish to represent in our int. policy
        
        '''
        def policy(state):
            
            for i, dbu in enumerate(stored_dbus):
                
                if dbu.is_point_in_DBU(state):
                    return actions[i]
            
            return actions[-1]
        
        return policy
    
    @staticmethod
    def tuplify_2D_array(two_d_array):
        two_d_list = []
        n,m = two_d_array.shape
        
        for i in range(n):
            two_d_list.append([])
            for j in range(m):
                two_d_list[-1].append(two_d_array[i,j])
        
            two_d_list[-1] = tuple(two_d_array[-1])                                 
        two_d_tuple = tuple(two_d_list)                                             
        return two_d_tuple                                                     
        
    
    def plot_errors(self):
        
        '''
        Plot the errors obtained after we perform the VIDTR algorithm.          

        '''
        
        for t in range(len(self.optimal_errors)):                                  
            plt.plot(np.arange(len(self.optimal_errors[t])), self.optimal_errors[t])
            plt.title(f'Errors at time {t}')
            plt.xlabel('Lengths')
            plt.ylabel('Errors')
            plt.show()
            