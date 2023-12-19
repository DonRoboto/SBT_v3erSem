

import os
import random
import argparse
import pandas as pd
from datetime import datetime
import numpy as np
import time
import csv

import gymnasium as gym

from ray.tune import run, sample_from
#from ray.tune.schedulers import PopulationBasedTraining
import myPBT
from ray.rllib.algorithms.algorithm import Algorithm


#Postprocess the perturbed config to ensure it's still valid used if PBT.
def explore(config):
    # Ensure we collect enough timesteps to do sgd.
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # Ensure we run at least one sgd iter.
    if config["lambda"] > 1:
        config["lambda"] = 1
    config["train_batch_size"] = int(config["train_batch_size"])
    return config


class Swarm():
    def __init__(self, ID, num_particles, bounds, check): 
        self.ID=ID
        self.num_particles = num_particles
        self.algo="PPO"
        self.env_name="LunarLanderContinuous-v2"
        self.timesteps_total=100000
        self.t_ready=50000
        self.perturb=0.25
        self.net="32_32"
        self.checkpoint=check        
        self.bounds=bounds
            
    def evaluate(self, generation):
        
        #pbt = PopulationBasedTraining(
        pbt = myPBT.SwarmBasedTraining(
            time_attr="timesteps_total",
            metric="episode_reward_mean",
            mode="max",
            perturbation_interval=self.t_ready,
            resample_probability=self.perturb,
            quantile_fraction=self.perturb,  # copy bottom % with top %
            # Specifies the search space for these hyperparams
            hyperparam_mutations={
                "lambda": lambda: random.uniform(self.bounds[0][0], self.bounds[0][1]),
                "clip_param": lambda: random.uniform(self.bounds[1][0], self.bounds[1][1]),
                "lr": lambda: random.uniform(self.bounds[2][0], self.bounds[2][1]),
                "train_batch_size": lambda: random.randint(self.bounds[3][0], self.bounds[3][1]),
            },
           custom_explore_fn=explore,
        )
        
        
        analysis = run(
            self.algo,
            name="test" + str(random.randint(100, 300)),
            scheduler=pbt,
            verbose=1,
            num_samples=self.num_particles,
            stop={"timesteps_total": self.timesteps_total * generation},
            config={
                "env": self.env_name,
                "log_level": "INFO",            
                #"seed": seed,
                "kl_coeff": 1.0,
                "num_gpus": 1,
                "horizon": 1600,
                "observation_filter": "MeanStdFilter",
                "model": {
                    "fcnet_hiddens": [
                        int(self.net.split("_")[0]),
                        int(self.net.split("_")[1]),
                    ],
                    "free_log_std": True,
                },
                "num_sgd_iter": 10,
                "sgd_minibatch_size": 128,
                "lambda": sample_from(lambda spec: random.uniform(self.bounds[0][0], self.bounds[0][1])),
                "clip_param": sample_from(lambda spec: random.uniform(self.bounds[1][0], self.bounds[1][1])),
                "lr": sample_from(lambda spec: random.uniform(self.bounds[2][0], self.bounds[2][1])),
                "train_batch_size": sample_from(lambda spec: random.randint(self.bounds[3][0], self.bounds[3][1])),
                
                #se agrega a la configuracion
                "ignore_worker_failures": True,
            },
            #se agrega a la configuracion
            restore=self.checkpoint,
            max_failures=100,
            reuse_actors=False,
            checkpoint_at_end=True,
        )
        
        
        best_conf = analysis.get_best_config(metric="episode_reward_mean", mode="max", scope="last")
        config_lambda  = best_conf['lambda']
        config_clip  = best_conf['clip_param']
        config_lr = best_conf['lr']
        config_batch = best_conf['train_batch_size']
        

        best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max", scope="last") 
        best_checkpoint = analysis.get_last_checkpoint(best_trial).to_directory()

        best_reward = best_trial.last_result['episode_reward_mean']
      
        
        #results to csv
        all_dfs = list(analysis.trial_dataframes.values())

        results = pd.DataFrame()
        for i in range(self.num_particles):
            df = all_dfs[i]
            df = df[
                [
                    "timesteps_total",
                    "episodes_total",
                    "episode_reward_mean",
                    "episode_reward_max",
                    "episode_reward_min",
                    "config/lambda",
                    "config/clip_param",                
                    "config/lr",                
                    "config/train_batch_size",                
                    "config/seed",
                ]
            ]
            df["Agent"] = i
            df["generation"] = generation
            results = pd.concat([results, df]).reset_index(drop=True)

    
        if not (os.path.exists("data/" )):
            os.makedirs("data/")

        results.to_csv("data/SBT_Swarm_{}.csv".format(str(self.ID)), mode="a")

        return best_checkpoint, best_reward, config_lambda, config_clip, config_lr, config_batch       


class Population():
    def __init__(self, num_swarms, n_part):
        self.num_swarms = num_swarms
        self.swarms = []

        self.bounds=[[0.9, 1.0], [0.1, 0.5], [1e-5, 1e-3], [1000, 60000]]        
        self.n_part = n_part
        
        for p in range(1, self.num_swarms+1):
            swarm = Swarm(ID=p, num_particles=n_part, bounds=self.bounds, check=None)
            self.swarms.append(swarm)
        
    def addSwarm(self, best_check):
        self.num_swarms += 1
        swarm = Swarm(ID=self.num_swarms, num_particles=self.n_part, bounds=self.bounds, check=best_check)
        self.swarms.append(swarm)
  
    def removeSwarm(self, worst_swarm):
        self.swarms.pop(worst_swarm)
        

population = Population(num_swarms=2, n_part=3)
bounds_0 = population.bounds


for g in range(1, 11):

    pop_checks = []
    pop_rewards = []

    for swarm in population.swarms:
        check, reward, config_0, config_1, config_2, config_3 = swarm.evaluate(generation=g)
        bounds=[[config_0*0.9, config_0*1.1], [config_1*0.9, config_1*1.1], [config_2*0.9, config_2*1.1], [int(config_3*0.9), int(config_3*1.1)]]
    
        for i in range(0,4):
            # adjust minimum position if necessary
            if bounds[i][1]<bounds_0[i][0]:
                bounds[i][0]=bounds_0[i][0]
                
            # adjust maximum position if necessary
            if bounds[i][1]>bounds_0[i][1]:
                bounds[i][1]=bounds_0[i][1]
    
        swarm.bounds=bounds
        swarm.checkpoint=check
        
        pop_checks.append(check)
        pop_rewards.append(reward)
        
        best_swarm = np.argmax(pop_rewards)
        best_check = pop_checks[best_swarm]
        worst_swarm = np.argmin(pop_rewards)
        
        
    
    print("######===========##########")
    print(pop_rewards)
    print("######===========##########")
    
    if g==3 or g==6:
        population.addSwarm(best_check)
        
    if g==5 or g==8:
        population.removeSwarm(worst_swarm)    

# #primera generacion    
# n_part = 5  
# bounds_0=[[0.9, 1.0], [0.1, 0.5], [1e-5, 1e-3], [1000, 60000]]
# s_1 = Swarm(num_particles=n_part, bounds=bounds_0, checkpoint=None, generation=1)
# check, config_0, config_1, config_2, config_3 = s_1.evaluate()    

# print(check)
# print(config_0)
# print(config_1)
# print(config_2)
# print(config_3)


# bounds=[[config_0*.9, config_0*1.1], [config_1*.9, config_1*1.1], [config_2*.9, config_2*1.1], [int(config_3*.9), int(config_3*1.1)]]
# bounds

# for i in range(0,4):
#     # adjust minimum position if neseccary
#     if bounds[i][1]<bounds_0[i][0]:
#         bounds[i][0]=bounds_0[i][0]
        
#     # adjust maximum position if necessary
#     if bounds[i][1]>bounds_0[i][1]:
#         bounds[i][1]=bounds_0[i][1]
# bounds
   

# #segunda generacion
# s_2 = Swarm(num_particles=n_part, bounds=bounds, checkpoint=check, generation=2)
# check, config_0, config_1, config_2, config_3 = s_2.evaluate()        

# print(check)
# print(config_0)
# print(config_1)
# print(config_2)
# print(config_3)


# bounds=[[config_0*.9, config_0*1.1], [config_1*.9, config_1*1.1], [config_2*.9, config_2*1.1], [int(config_3*.9), int(config_3*1.1)]]
# bounds

# for i in range(0,4):
#     # adjust minimum position if neseccary
#     if bounds[i][1]<bounds_0[i][0]:
#         bounds[i][0]=bounds_0[i][0]
        
#     # adjust maximum position if necessary
#     if bounds[i][1]>bounds_0[i][1]:
#         bounds[i][1]=bounds_0[i][1]
# bounds
   
# #tercera genreacion
# s_3 = Swarm(num_particles=n_part, bounds=bounds, checkpoint=check, generation=3)
# check, config_0, config_1, config_2, config_3 = s_3.evaluate()        

# print(check)
# print(config_0)
# print(config_1)
# print(config_2)
# print(config_3)

# bounds=[[config_0*.9, config_0*1.1], [config_1*.9, config_1*1.1], [config_2*.9, config_2*1.1], [int(config_3*.9), int(config_3*1.1)]]
# bounds

# for i in range(0,4):
#     # adjust minimum position if neseccary
#     if bounds[i][1]<bounds_0[i][0]:
#         bounds[i][0]=bounds_0[i][0]
        
#     # adjust maximum position if necessary
#     if bounds[i][1]>bounds_0[i][1]:
#         bounds[i][1]=bounds_0[i][1]
# bounds



# #cuarta genreacion
# s_4 = Swarm(num_particles=n_part, bounds=bounds, checkpoint=check, generation=4)
# check, config_0, config_1, config_2, config_3 = s_4.evaluate()        

# print(check)
# print(config_0)
# print(config_1)
# print(config_2)
# print(config_3)

# bounds=[[config_0*.9, config_0*1.1], [config_1*.9, config_1*1.1], [config_2*.9, config_2*1.1], [int(config_3*.9), int(config_3*1.1)]]
# bounds

# for i in range(0,4):
#     # adjust minimum position if neseccary
#     if bounds[i][1]<bounds_0[i][0]:
#         bounds[i][0]=bounds_0[i][0]
        
#     # adjust maximum position if necessary
#     if bounds[i][1]>bounds_0[i][1]:
#         bounds[i][1]=bounds_0[i][1]
# bounds



# #quinta genreacion
# s_5 = Swarm(num_particles=n_part, bounds=bounds, checkpoint=check, generation=5)
# check, config_0, config_1, config_2, config_3 = s_5.evaluate()        

# print(check)
# print(config_0)
# print(config_1)
# print(config_2)
# print(config_3)

# bounds=[[config_0*.9, config_0*1.1], [config_1*.9, config_1*1.1], [config_2*.9, config_2*1.1], [int(config_3*.9), int(config_3*1.1)]]
# bounds

# for i in range(0,4):
#     # adjust minimum position if neseccary
#     if bounds[i][1]<bounds_0[i][0]:
#         bounds[i][0]=bounds_0[i][0]
        
#     # adjust maximum position if necessary
#     if bounds[i][1]>bounds_0[i][1]:
#         bounds[i][1]=bounds_0[i][1]
# bounds