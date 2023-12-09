
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
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.algorithm import Algorithm



# Postprocess the perturbed config to ensure it's still valid used if PBT.
def explore(config):
    # Ensure we collect enough timesteps to do sgd.
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # Ensure we run at least one sgd iter.
    if config["lambda"] > 1:
        config["lambda"] = 1
    config["train_batch_size"] = int(config["train_batch_size"])
    return config




num_samples = 20
num_workers = 4
perturb = 0.25
algo = "PPO"
seed = 1721
timesteps_total = 1000000
t_ready = 50000

#env_name = "LunarLanderContinuous-v2"
env_name = "BipedalWalker-v3"
#env_name = "Pendulum-v1"

#Acrobot requiere "free_log_std": False,
#env_name = "Acrobot-v1"


horizon = 1600
method = "pbt"

filename = "02"
save_csv = True

net = "32_32"

test_episodes = 100

timelog = (
    str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now()))
)

directory = "{}_{}_{}_Size{}_{}".format(
    algo,
    filename,
    method,
    str(num_samples),
    env_name,
)


pbt = PopulationBasedTraining(
    time_attr="timesteps_total",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=t_ready,
    resample_probability=perturb,
    quantile_fraction=perturb,  # copy bottom % with top %
    # Specifies the search space for these hyperparams
    hyperparam_mutations={
        "lambda": lambda: random.uniform(0.9, 1.0),
        "clip_param": lambda: random.uniform(0.1, 0.5),
        "lr": lambda: random.uniform(1e-5, 1e-3),
        "train_batch_size": lambda: random.randint(1000, 60000),
    },
    custom_explore_fn=explore,

)

timelog = (
    str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now()))
)


##########
###TRAIN##
##########

time_start = time.time()

analysis = run(
    algo,
    name="{}_{}_{}_seed{}_{}".format(
        timelog, method, env_name, str(seed), filename
    ),
    scheduler=pbt,
    verbose=1,
    num_samples=num_samples,
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
        "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
        "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
        "lr": sample_from(lambda spec: random.uniform(1e-5, 1e-3)),
        "train_batch_size": sample_from(lambda spec: random.randint(1000, 60000)),
        
        #se agrega a la configuracion
        "ignore_worker_failures": True,
    },
    #se agrega a la configuracion
    max_failures=100,
    reuse_actors=False,
    checkpoint_at_end=True,
)

all_dfs = list(analysis.trial_dataframes.values())

results = pd.DataFrame()
for i in range(num_samples):
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

if save_csv:
    if not (os.path.exists("data/" + directory)):
        os.makedirs("data/" + directory)

    results.to_csv("data/{}/seed{}.csv".format(directory, str(seed)))


time_end = time.time()
print(time_end - time_start)


#########
###TEST##
#########
best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max") 
checkpoint = best_trial.checkpoint


best_config = best_trial.config

loaded_ppo = Algorithm.from_checkpoint(best_trial.checkpoint)
loaded_policy = loaded_ppo.get_policy()


# See trained policy in action
env = gym.make(env_name)


test_rows=[]

for i in range(test_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:        
        action = loaded_policy.compute_single_action(np.array(state))
        state, reward, done, _, _ = env.step(action[0])
        total_reward += reward

        steps += 1

    test_rows.append([i, steps, total_reward])
    
print(test_rows)

fields = ['ID', 'steps', 'total_reward']
with open('test_' + str(env_name) + '_'+ str(method) + '_'  + str(seed) + '.csv', 'w') as f:         
    write = csv.writer(f)
     
    write.writerow(fields)
    write.writerows(test_rows)


with open('conf_' + str(env_name) + '_' + str(method) + '_' + str(seed) + '.txt', 'w') as f:
    f.write(str(best_config))
    f.write('\n\nTIME: ' + str(time_end - time_start))

