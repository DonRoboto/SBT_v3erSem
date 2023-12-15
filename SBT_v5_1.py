
#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray import tune
#from ray import air
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

from datetime import datetime

import time
from csv import writer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import random

import numpy as np
from csv import writer

import transformations as trafo
import itertools
import math

import pandas as pd

from ray.tune import run
    
class Population:
    def __init__(self, num_swarms, num_particles, bounds):
        self.swarms=[]
        self.num_swarms = num_swarms
        self.best_check = None
        self.err_best_g = -1000000
        self.s_reward = list(np.zeros(num_swarms))
        
        total_pop = self.num_swarms * num_particles
        
        sampling = LHS(xlimits=np.array(bounds))
        X = sampling(total_pop)
        print(X)

        k=1
        for i in range(0, total_pop, num_particles):            
            s = Swarm(X[i:(num_particles*k)], num_particles, i)
            self.swarms.append(s)
            k+=1
            
            
    def addSwarm(self):
        self.num_swarms += 1
        sampling = LHS(xlimits=np.array(bounds))
        X = sampling(num_particles)
            
        s = Swarm(X, num_particles, self.num_swarms-1)
        s.best_check = self.best_check
        
        for part in s.particles:
            part.model = part.ppo_config.build()
            part.model.load_checkpoint(self.best_check)  
            
        self.swarms.append(s)
        self.s_reward.append(0)
        

    def removeSwarm(self):
        print("##worst_swarm")
        max_index = np.argmax(self.s_reward)
        min_index = np.argmin(self.s_reward)
        
        self.s_reward[min_index] = self.s_reward[max_index]
        self.swarms[min_index].particles = self.swarms[max_index].particles.copy()
        self.swarms[min_index].ID = random.randint(100, 200)


class Swarm:
    def __init__(self, X, num_particles, ID):
        print("#$$$$$")
        print(X)
        self.ID=ID
        self.particles=[]    
        self.converge=False
        
        self.err_best_g=-1000000               # best error for group
        self.pos_best_g=[]                     # best position for group        
        self.best_check=None
        
        for i in range(0, num_particles):
            self.particles.append(Particle(X[i], i))
 
    
    def test_converge(self):
        rexcl=0.01
        pos = []        
        for part in self.particles:
            pos.append(part.position_i)

        #estandarizar posiciones
        pos_std = trafo.unit_vector(pos, axis=0)    
        print(pos)
        print(pos_std)
        
        for p1, p2 in itertools.combinations(pos_std, 2):
            print(p1)
            print(p2)
            
            d = math.sqrt(sum((x1 - x2)**2. for x1, x2 in zip(p1, p2)))
            print(d)
            
            if d < rexcl:                
                print("si conv")
                break
            
        return self.converge  
    
        #     if d > rexcl:
        #         self.converge=False
        #         print("not conv")
        #     else:
        #         self.converge=True
        #         print("si conv")
                
        # return self.converge       


    def convertQuantum(self, bounds):
        centre = self.pos_best_g
        dim = 4
        for part in self.particles:
            for i in range(0, dim):   
                part.position_i[i] = centre[i] * random.choice([0.8, 1.2])
                
                # adjust minimum position if neseccary
                if part.position_i[i]<bounds[i][0]:
                    part.position_i[i]=bounds[i][0]
                
                # adjust maximum position if necessary
                if part.position_i[i]>bounds[i][1]:
                    part.position_i[i]=bounds[i][1]
                

class Particle:
    def __init__(self, x0, ID):
        self.ID=ID                  # particle ID
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1000000          # best error individual
        self.err_i=-1000000              # error individual
        self.checkpoint=None

        
        # #self.env_name = 'CartPole-v1'
        # self.env_name = 'LunarLanderContinuous-v2'
        # #self.env_name = 'BipedalWalker-v3'
        # #self.env_name = 'Pendulum-v1'
        
        self.lambda_= x0[0]
        self.lr = x0[1]
        self.clip_param = x0[2]
        self.train_batch_size = int(x0[3])
        
        # self.ppo_config = PPOConfig().environment(self.env_name)
        # self.ppo_config.stop={"timesteps_total": 1000000}
        # #self.ppo_config.timesteps_total=500
        
        # # #self.ppo_config.resources_per_trial={"gpu": 0.1}
        
        # self.ppo_config.fault_tolerance(recreate_failed_workers=False, 
        #                                 num_consecutive_worker_failures_tolerance=1000)
        
        # # self.ppo_config.recreate_failed_workers=False
        # # self.ppo_config.max_num_worker_restarts = 1000
        # # self.ppo_config.delay_between_worker_restarts_s = 60.0
        # # self.ppo_config.restart_failed_sub_environments = False
        # # self.ppo_config.num_consecutive_worker_failures_tolerance = 100
        # # self.ppo_config.worker_health_probe_timeout_s = 60
        # # self.ppo_config.worker_restore_timeout_s = 1800
        # # self.ppo_config.create_env_on_driver=True
        
        # self.ppo_config.horizon=1600
        # self.ppo_config.observation_filter="MeanStdFilter"
        # self.ppo_config.fcnet_hiddens=[32, 32]
        # self.ppo_config.free_log_std=True
        # # config={
        # # "env": env_name,
        # # "log_level": "INFO",            
        # # #"seed": seed,
        # # "kl_coeff": 1.0,
        # # "num_gpus": 1,
        # # "horizon": horizon,
        # # "observation_filter": "",
        # # "model": {
        # #     "fcnet_hiddens": [
        # #         int(net.split("_")[0]),
        # #         int(net.split("_")[1]),
        # #     ],
        # #     "free_log_std": True,
        # # }

        # #self.ppo_config.resources(num_gpus=1)
        # # self.ppo_config.rollouts(num_rollout_workers=4)
        
        # self.ppo_config.reuse_actors=True
        # self.ppo_config.ignore_worker_failures=True
        # self.ppo_config.max_failures=1000
        
        # self.ppo_config.workers=1
        
        # #self.ppo_config.resources_per_trial={"cpu": 0.01},
        
                     
        # self.ppo_config.training(      
        #   lambda_=self.lambda_,
        #   lr=self.lr,
        #   clip_param=self.clip_param,
        #   train_batch_size=self.train_batch_size
        # )  
        
        # self.model = self.ppo_config.build()

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-0.001, 0.001))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, generation, id_swarm):
        #try:
        self.err_i = self.costFunc(generation, id_swarm)
        #except:
        #    print("###====EXCEPTION TRAIN====####")

        # check to see if the current position is an individual best
        if self.err_i<self.err_best_i or self.err_best_i==-1000000:
            self.pos_best_i=self.position_i.copy()
            self.err_best_i=self.err_i
 
    
    def costFunc(self, generation, id_swarm):
        algo = "PPO"
        method = "sbt"
        env_name = "LunarLanderContinuous-v2"
        seed = 1
        filename = "02"
        timesteps_total = 100000
        horizon = 1600
        net = "32_32"
        timelog = (
            str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now()))
        )
        ###TRAIN#### 
        # t_error=False
        time_start = time.time()   
        
        
        analysis = run(
            algo,
            name="{}_{}_{}_seed{}_{}".format(
                timelog, method, env_name, str(seed), filename
            ),
            #scheduler=pbt,
            verbose=1,
            num_samples=1,
            stop={"timesteps_total": timesteps_total},
            config={
                "env": env_name,
                "log_level": "INFO",            
                #"seed": seed,
                "kl_coeff": 1.0,
                "num_gpus": 1,
                "horizon": horizon,
                "observation_filter": "MeanStdFilter",
                "model": {
                    "fcnet_hiddens": [
                        int(net.split("_")[0]),
                        int(net.split("_")[1]),
                    ],
                    "free_log_std": True,
                },
                "num_sgd_iter": 10,
                "sgd_minibatch_size": 128,
                #PARAMS
                "lambda": self.lambda_,
                "clip_param": self.clip_param,
                "lr": self.lr,
                "train_batch_size": self.train_batch_size,
                
                #se agrega a la configuracion
                "ignore_worker_failures": True,
            },
            
            restore=self.checkpoint,
            
            #se agrega a la configuracion
            max_failures=100,
            reuse_actors=False,
            checkpoint_at_end=True,
        )

 
        time_end = time.time()
        print("TIME:")
        print(time_end - time_start)
        
        #save_result = self.model.save()
        path_to_checkpoint = analysis.get_best_checkpoint(trial=analysis.get_best_trial(), metric="episode_reward_mean", mode="max")
        self.checkpoint = path_to_checkpoint.to_directory()
        #self.model.stop()
        #self.model.cleanup()
        
        
        all_dfs = list(analysis.trial_dataframes.values())
        
        results = pd.DataFrame()
        for i in range(1):
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
            results = pd.concat([results, df]).reset_index(drop=True)
        
        df_reward = df['episode_reward_mean']
        df_timesteps = df['timesteps_total']
        df_episodes = df['episodes_total']
        
        reward = df_reward[-1:].values[0]
        time_steps = df_timesteps[-1:].values[0]
        episodes = df_episodes[-1:].values[0]
        
        
        print("####config######")
        print("lambda_:", self.lambda_)
        print("lr:", self.clip_param)
        print("clip_param:", self.lr)
        print("train_batch_size:", self.train_batch_size)
        print("reward:", reward)
        
        print("####config######")
        
        data_csv = [self.lambda_,
                    self.clip_param,
                    self.lr,
                    self.train_batch_size,  
                    self.ID,
                    id_swarm,                    
                    generation,                    
                    time_steps,
                    episodes,
                    reward
                    #train['hist_stats']['episode_reward'],
                    #train['hist_stats']['episode_lengths']
                    ]
        
        print("CSV")
        print(data_csv)
        print("PPOCONFIG")
        print(self.lambda_)
        print(self.clip_param)
        print(self.lr)
        print(self.train_batch_size)
        print("PPOCONFIG")
        
        with open('event.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(data_csv)         
            f_object.close()
        
            #return policy, path_to_checkpoint, np.mean(train['hist_stats']['episode_reward'])
        # else:
        #     reward=-1000000
        return reward

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w=0.2       # constant inertia weight (how much to weigh the previous velocity)
        c1=0.3        # cognative constant
        c2=0.6        # social constant
        
        for i in range(0,num_dimensions):
            r1=random.uniform(0.1, 0.9)
            r2=random.uniform(0.1, 0.9)
            
            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds, checkpoint):
        print("load check")
        print(checkpoint)
        self.checkpoint=checkpoint
        print("load check")
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            
            # adjust minimum position if neseccary
            if self.position_i[i]<bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

        # self.model = self.ppo_config.build()
        # self.model.load_checkpoint(checkpoint)   
        
        self.lambda_= self.position_i[0]
        self.lr = self.position_i[1]
        self.clip_param = self.position_i[2]
        self.train_batch_size = int(self.position_i[3])
        
        print("load checkpoint")
       
        # self.ppo_config.training(      
        #   lambda_= self.position_i[0],
        #   lr=self.position_i[1],
        #   clip_param=self.position_i[2],
        #   train_batch_size=int(self.position_i[3])
        # )  
        
num_dimensions=4
num_swarms=5
num_particles=4
generations=10
bounds=[[0.9, 1.0], [1e-5, 1e-3], [0.1, 0.5], [1000, 60000]]

population = Population(num_swarms=num_swarms, num_particles=num_particles, bounds=bounds)

for g in range(generations):
    
    #agregar/quitar swarms
    if g==3 or g==6: 
        population.removeSwarm()
    
    # if g==5 or g==7:
    #     population.removeSwarm()
        
    # if g==4 or g==6 or g==9:
    #     population.addSwarm()
    
    k=0
    for swarm in population.swarms:
           
        # cycle through particles in swarm and evaluate fitness
        for part in swarm.particles:
            part.evaluate(g, swarm.ID)
            

        # determine if current particle is the best (globally)        
        swarm.err_best_g=-10000000
        for part in swarm.particles:                
            if part.err_i>swarm.err_best_g:
                swarm.pos_best_g=list(part.position_i.copy())
                swarm.err_best_g=float(part.err_i)
                swarm.best_check = part.checkpoint
                
                population.s_reward[k]=swarm.err_best_g
                
            if swarm.err_best_g>population.err_best_g:
                population.err_best_g=swarm.err_best_g
                population.best_check = swarm.best_check
        
        
        #verifica convergencia
        # test_conv = swarm.test_converge()
        # if test_conv:
        #     swarm.convertQuantum(bounds)
        

        # cycle through swarm and update velocities and position        
        for part in swarm.particles:
            part.update_velocity(swarm.pos_best_g)
            part.update_position(bounds, swarm.best_check)
        

        print("====####S_REWARD###====")
        print(population.s_reward)        
        k+=1
        #graph swarm
        #graph(swarm, [pos_best_g], i)
        #err_best_g=-10000
        #print(swarm.best_check)
        


# p1 = population.swarms[0].particles[1].position_i[2]
# print(p1)


# p2 = str(population.swarms[0].particles[1].lr)
# print(p2)


# population.swarms[0].particles = population.swarms[1].particles.copy()

# population.swarms[3].particles[2].model.create_env_on_driver=True
# e = population.swarms[3].particles[2].model.evaluate(10)


# m = population.swarms[3].particles[2].model.evaluate(10)
# m


# c = population.swarms[1].particles[0].ppo_config.fault_tolerance()
# c.fault_tolerance

# t = population.swarms[1].particles[0].model.train()
# t.
# s = t.get_state()
# ms = s['module_state']['default_policy']
# ms




# b = population.swarms[1].particles[0].model.config
# b.evaluation_num_workers
# b

# t = population.swarms[1].particles[0].ppo_config.build()



# for part in population.swarms[1].particles:
#     policy = part.model.get_default_policy_class(part.model.config) 
#     print(policy.



# #
# #verifica convergencia
# #
# import itertools
# import math
    
# population.swarms[0].particles[0].position_i

# rexcl=40000

# for p1, p2 in itertools.combinations(population.swarms[0].particles, 2):
#     print(p1.position_i)
#     print(p2.position_i)
#     #x1 = p1.position_i
#     #x2 = p2.position_i
#     print("##########")
    
#     d = math.sqrt(sum((x1 - x2)**2. for x1, x2 in zip(p1.position_i, p2.position_i)))
#     print(d)
#     if d > 2*rexcl:
#         print("not conv")
#     else:
#         print("si conv")


# import transformations as trafo
# import numpy as np

# data = np.array([[0.975, 0.0002575, 0.2, 15750],
#  [0.925,	0.0007525,	0.4,	45250],
#   [0.975,	0.0002575,	0.4,	45250],
#    [0.925,	0.0007525,	0.2,	15750]])

# print(trafo.unit_vector(data, axis=0))

# [[0.5129803  0.22893411 0.31622777 0.23244226]
#  [0.48667362 0.66902105 0.63245553 0.6678103 ]
#  [0.5129803  0.22893411 0.63245553 0.6678103 ]
#  [0.48667362 0.66902105 0.31622777 0.23244226]]


# train = population.swarms[1].particles[0].model.train()
# print(train[''])
# print(train['info']['learner']['__all__'][])
# cols = train.columns