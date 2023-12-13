
#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray import tune
#from ray import air
import matplotlib.pyplot as plt

from smt.sampling_methods import LHS

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

    
class Population:
    def __init__(self, num_swarms, num_particles, bounds):
        self.swarms=[]
        self.num_swarms = num_swarms
        self.best_check = None
        self.err_best_g = -1000000
        self.s_reward = list(np.zeros(num_swarms))
        
        for i in range(0, num_swarms):            
            sampling = LHS(xlimits=np.array(bounds))
            X = sampling(num_particles)
            
            s = Swarm(X, num_particles, i)
            self.swarms.append(s)
            
            
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
        r_index = np.argmin(self.s_reward)
        
        self.s_reward.pop(r_index)
        self.swarms.pop(r_index)


class Swarm:
    def __init__(self, X, num_particles, ID):
        self.ID=ID
        self.particles=[]    
        self.converge=False
        
        self.err_best_g=-1000000               # best error for group
        self.pos_best_g=[]                     # best position for group        
        self.best_check=None
        
        for i in range(0, num_particles):
            self.particles.append(Particle(X[i], i))
 
    
    def test_converge(self):
        rexcl=0.2
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
            if d > rexcl:
                self.converge=False
                print("not conv")
            else:
                self.converge=True
                print("si conv")
                
        return self.converge       


    def convertQuantum(self, bounds):
        centre = self.pos_best_g
       # RCLOUD = 1   
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
                
            # position = [gauss(0, 1) for _ in range(dim)]
            # dist = math.sqrt(sum(x**2 for x in position))
    
            # # if dist == "gaussian":
            # u = abs(gauss(0, 1.0/3.0))
            # part.position_i = [(RCLOUD * x * u**(1.0/dim) / dist) + c for x, c in zip(position, centre)]
                     
            # elif dist == "uvd":
            #     u = random.random()
            #     part.position_i = [(RCLOUD * x * u**(1.0/dim) / dist) + c for x, c in zip(position, centre)]
    
            # elif dist == "nuvd":
            #     u = abs(random.gauss(0, 1.0/3.0))
            #     part.position_i = [(RCLOUD * x * u / dist) + c for x, c in zip(position, centre)]
    
            # del part.fitness.values
            # del part.bestfit.values
            # part.best = None
            
    
        #return swarm



class Particle:
    def __init__(self, x0, ID):
        self.ID=ID                  # particle ID
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        self.checkpoint=None

        
        #self.env_name = 'CartPole-v1'
        self.env_name = 'LunarLanderContinuous-v2'
        #self.env_name = 'BipedalWalker-v3'
        #self.env_name = 'Pendulum-v1'
        
        self.lambda_= x0[0]
        self.lr = x0[1]
        self.clip_param = x0[2]
        self.train_batch_size = int(x0[3])
        
        self.ppo_config = PPOConfig().environment(self.env_name)
        self.ppo_config.stop={"timesteps_total": 500}
        #self.ppo_config.timesteps_total=500
        
        # #self.ppo_config.resources_per_trial={"gpu": 0.1}
        # #self.ppo_config.num_gpus=1
        
        # self.ppo_config.ignore_worker_failures=True
        # self.ppo_config.recreate_failed_workers=False
        # self.ppo_config.max_num_worker_restarts = 1000
        # self.ppo_config.delay_between_worker_restarts_s = 60.0
        # self.ppo_config.restart_failed_sub_environments = False
        # self.ppo_config.num_consecutive_worker_failures_tolerance = 100
        # self.ppo_config.worker_health_probe_timeout_s = 60
        # self.ppo_config.worker_restore_timeout_s = 1800
        # self.ppo_config.create_env_on_driver=True

        self.ppo_config.resources(num_gpus=1)  
        # self.ppo_config.rollouts(num_rollout_workers=4)
        
        self.ppo_config.reuse_actors=False
        self.ppo_config.ignore_worker_failures=True
        self.ppo_config.max_failures=1000
        
                     
        self.ppo_config.training(      
          lambda_=self.lambda_,
          lr=self.lr,
          clip_param=self.clip_param,
          train_batch_size=self.train_batch_size
        )  
        
        self.model = self.ppo_config.build()

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-0.001, 0.001))
            self.position_i.append(x0[i])
            #self.position_i.append(uniform(bounds[i][0], bounds[i][1]))

    # evaluate current fitness
    def evaluate(self, generation, id_swarm):
        self.err_i = self.costFunc(generation, id_swarm)

        # check to see if the current position is an individual best
        if self.err_i<self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i.copy()
            self.err_best_i=self.err_i
 
    
    def costFunc(self, generation, id_swarm):
        
        #self.model = self.ppo_config.build()
        
        ###TRAIN####    
        time_start = time.time()    
        train = self.model.train()
        time_end = time.time()
        print("TIME:")
        print(time_end - time_start)
        
        save_result = self.model.save()
        path_to_checkpoint = save_result.checkpoint.path        
        self.checkpoint = path_to_checkpoint
        self.model.stop()
        
        reward = np.mean(train['hist_stats']['episode_reward'])
        
        print("####config######")
        print("lambda_:", self.position_i[0])
        print("lr:", self.position_i[1])
        print("clip_param:", self.position_i[2])
        print("train_batch_size:", self.position_i[3])
        print("reward:", reward)
        
        print("####config######")
        
        data_csv = [train['num_agent_steps_sampled'],
                    len(train['hist_stats']['episode_lengths']),
                    train['episode_reward_mean'],
                    train['episode_reward_max'],
                    train['episode_reward_min'],
                    self.ppo_config.lambda_,
                    self.ppo_config.clip_param,
                    self.ppo_config.lr,
                    self.ppo_config.train_batch_size,
                    '0',
                    self.ID,
                    id_swarm,                    
                    generation,
                    path_to_checkpoint
                    #train['hist_stats']['episode_reward'],
                    #train['hist_stats']['episode_lengths']
                    ]
        
        print("CSV")
        print(data_csv)
        print("PPOCONFIG")
        print(self.ppo_config.lambda_)
        print(self.ppo_config.clip_param)
        print(self.ppo_config.lr)
        print(self.ppo_config.train_batch_size)
        print("PPOCONFIG")
        
        with open('event.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(data_csv)         
            f_object.close()
    
        #return policy, path_to_checkpoint, np.mean(train['hist_stats']['episode_reward'])
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
        print("load check")
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            
            # adjust minimum position if neseccary
            if self.position_i[i]<bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

        self.model = self.ppo_config.build()
        self.model.load_checkpoint(checkpoint)   
        print("load checkpoint")
       
        self.ppo_config.training(      
          lambda_= self.position_i[0],
          lr=self.position_i[1],
          clip_param=self.position_i[2],
          train_batch_size=int(self.position_i[3])
        )  
        
num_dimensions=4
num_swarms=2
num_particles=2
generations=5
bounds=[[0.9, 1.0], [1e-5, 1e-3], [0.1, 0.5], [1000, 60000]]

population = Population(num_swarms=num_swarms, num_particles=num_particles, bounds=bounds)

for g in range(generations):
    
    #agregar/quitar swarms
    if g==2:
        population.addSwarm()
    
    if g==4:
        population.removeSwarm()
    
    k=0
    for swarm in population.swarms:
           
        # cycle through particles in swarm and evaluate fitness
        #for j in range(0, num_particles):
        for part in swarm.particles:
            #swarm.particles[j].evaluate(g)             
            part.evaluate(g, swarm.ID)
            

        # determine if current particle is the best (globally)
        #for j in range(0, num_particles): 
        for part in swarm.particles:    
            #if swarm.particles[j].err_i>err_best_g:
            if part.err_i>swarm.err_best_g:
                # pos_best_g=list(swarm.particles[j].position_i)
                # err_best_g=float(swarm.particles[j].err_i)
                # best_check = swarm.particles[j].checkpoint
                swarm.pos_best_g=list(part.position_i.copy())
                swarm.err_best_g=float(part.err_i)
                swarm.best_check = part.checkpoint
                
                population.s_reward[k]=swarm.err_best_g
                
            if swarm.err_best_g>population.err_best_g:
                population.err_best_g=swarm.err_best_g
                population.best_check = swarm.best_check
        
        
        #verifica convergencia
        print("Test convergencia")
        # for part in swarm.particles:
        #     print(part.position_i)
        
        test_conv = swarm.test_converge()
        if test_conv:
            swarm.convertQuantum(bounds)
        
        # for part in swarm.particles:
        #     print(part.position_i)
        
        # cycle through swarm and update velocities and position
        #for j in range(0, num_particles):
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