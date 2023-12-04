

import os
import random
import argparse
import pandas as pd
from datetime import datetime

from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining
#from ray.tune.schedulers.pb2 import PB2

import numpy as np
import time

#star time
start_time = time.time()


parser = argparse.ArgumentParser()
#parser.add_argument("--max", type=int, default=1000000)
#parser.add_argument("--max", type=int, default=1000000)
parser.add_argument("--algo", type=str, default="PPO")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--num_samples", type=int, default=20)
#parser.add_argument("--t_ready", type=int, default=50000)
parser.add_argument("--t_ready", type=int, default=500000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    #"--horizon", type=int, default=200
    "--horizon", type=int, default=1600
)  # make this 1000 for other envs
parser.add_argument("--perturb", type=float, default=0.25)  # if using PBT
#parser.add_argument("--env_name", type=str, default="BipedalWalker-v3")
parser.add_argument("--env_name", type=str, default="LunarLanderContinuous-v2")
#parser.add_argument("--env_name", type=str, default="MountainCarContinuous-v0")
#parser.add_argument("--env_name", type=str, default="CartPole-v1")


parser.add_argument(
    "--criteria", type=str, default="timesteps_total"
)  # "training_iteration", "time_total_s"
parser.add_argument(
    "--net", type=str, default="32_32"
)  # May be important to use a larger network for bigger tasks.
parser.add_argument("--filename", type=str, default="")
parser.add_argument("--method", type=str, default="pbt")  # ['pbt', 'pb2']
parser.add_argument("--save_csv", type=bool, default=False)

args = parser.parse_args()

# bipedalwalker needs 1600
if args.env_name in ["LunarLanderContinuous-v2", "MountainCarContinuous-v0", "CartPole-v1"]:
    horizon = 150
else:
    horizon = 150

pbt = PopulationBasedTraining(
    time_attr=args.criteria,
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=args.t_ready,
    resample_probability=args.perturb,
    quantile_fraction=args.perturb,  # copy bottom % with top %
    # Specifies the search space for these hyperparams
    hyperparam_mutations={
        #"lambda": lambda: random.uniform(0.9, 1.0),
        #"clip_param": lambda: random.uniform(0.1, 0.5),
        "clip_param": lambda: random.uniform(0.1, 0.5),
        #"lr": lambda: random.uniform(1e-5, 1e-3),
        #"lr": lambda: random.uniform(1e-5, 1e-3),
        "lr": lambda: random.uniform(1e-5, 1e-3),
        #"train_batch_size": lambda: random.randint(1000, 60000),
        
    },
   # custom_explore_fn=explore,
)

# pb2 = PB2(
#     time_attr=args.criteria,
#     metric="episode_reward_mean",
#     mode="max",
#     perturbation_interval=args.t_ready,
#     quantile_fraction=args.perturb,  # copy bottom % with top %
#     # Specifies the hyperparam search space
#     hyperparam_bounds={
#         "lambda": [0.9, 1.0],
#         "clip_param": [0.1, 0.5],
#         "lr": [1e-5, 1e-3],
#         "train_batch_size": [1000, 60000],
#     },
# )

methods = {"pbt": pbt}

timelog = (
    str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now()))
)

args.dir = "{}_{}_{}_Size{}_{}_{}".format(
    args.algo,
    args.filename,
    args.method,
    str(args.num_samples),
    args.env_name,
    args.criteria,
)

analysis = run(
    args.algo,
    name="{}_{}_{}_seed{}_{}".format(
        timelog, args.method, args.env_name, str(args.seed), args.filename
    ),
    scheduler=methods[args.method],
    verbose=1,
    num_samples=args.num_samples,
    max_failures=100,
    #stop={args.criteria: args.max},
    stop={"timesteps_total": 500000},
    config={
        "env": args.env_name,
        "log_level": "INFO",
        "ignore_worker_failures": True,
        #"seed": args.seed,
        #"kl_coeff": 1.0,
        "num_gpus": 1,
        #"horizon": horizon,
        "observation_filter": "MeanStdFilter",
        "model": {
            "fcnet_hiddens": [
                int(args.net.split("_")[0]),
                int(args.net.split("_")[1]),
            ],
            "free_log_std": False,
        },
        "num_sgd_iter": 10,
        #"sgd_minibatch_size": 128,
        #"lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
        "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
        "lr": sample_from(lambda spec: random.uniform(1e-5, 1e-3)),
        #"train_batch_size": sample_from(lambda spec: random.randint(1000, 60000)),
    },
)

all_dfs = list(analysis.trial_dataframes.values())

print("#####")
end_time = (time.time() - start_time)
print("Process time: " + str(end_time))
print("#####")
    

best_df = analysis.best_trial

# df_0 = all_dfs[0]
# df_time_steps_0 = df_0['agent_timesteps_total']
# df_episode_reward_mean_0 = df_0['episode_reward_mean']
# df_clip_param_0 = df_0['config/clip_param']
# df_lr_0 = df_0['config/lr']
# df_hist_reward_0 = df_0['hist_stats/episode_reward']
# df_hist_lengths_0 = df_0['hist_stats/episode_lengths']
# df_episodes_total_0 = df_0['episodes_total']



# df_0 = all_dfs[5]
# df_time_steps_0 = df_0[['agent_timesteps_total', 'episode_reward_mean','config/clip_param','config/lr','episodes_total','hist_stats/episode_reward','hist_stats/episode_lengths']]

dfs_0 = pd.DataFrame()
for i in range(20):
    df_aux = all_dfs[i]
    dff = df_aux[['episode_reward_mean']]                    
    dfs_0 = pd.concat([dfs_0, dff],axis = 1)

dfs_1 = pd.DataFrame()
for i in range(20):
    df_aux = all_dfs[i]
    dff = df_aux[['agent_timesteps_total']]                    
    dfs_1 = pd.concat([dfs_1, dff],axis = 1)

                      
dfs_2 = pd.DataFrame()
for i in range(20):
    df_aux = all_dfs[i]
    dff = df_aux[['config/clip_param']]                    
    dfs_2 = pd.concat([dfs_2, dff],axis = 1)

                      
dfs_3 = pd.DataFrame()
for i in range(20):
    df_aux = all_dfs[i]
    dff = df_aux[['config/lr']]                    
    dfs_3 = pd.concat([dfs_3, dff],axis = 1)

            
            
dfs_4 = pd.DataFrame()
for i in range(20):
    df_aux = all_dfs[i]
    dff = df_aux[['hist_stats/episode_reward']]                    
    dfs_4 = pd.concat([dfs_4, dff],axis = 1)


dfs_5 = pd.DataFrame()
for i in range(20):
    df_aux = all_dfs[i]
    dff = df_aux[['hist_stats/episode_lengths']]                    
    dfs_5 = pd.concat([dfs_5, dff],axis = 1)

            
dfs_6 = pd.DataFrame()
for i in range(20):
    df_aux = all_dfs[i]
    dff = df_aux[['episodes_total']]                    
    dfs_6 = pd.concat([dfs_6, dff],axis = 1)


colum = df_aux.columns
            





z_test1 = df_aux[['config/lr_schedule']]   
z_test2 = df_aux[['']]   

    #dfs.append(df_time_steps_0)
    #dfs = np.concatenate((dfs, dff), axis=0)
    #dfs = str(dfs) + str(df_time_steps_0)

    
#dff = pd.concat(dfs, ignore_index=True)

# df_1 = all_dfs[1]
# df_time_steps_1 = df_1[['agent_timesteps_total', 'episode_reward_mean','config/clip_param','config/lr','episodes_total','hist_stats/episode_reward','hist_stats/episode_lengths']]


# df_2 = all_dfs[2]
# df_time_steps_2 = df_2[['agent_timesteps_total', 'episode_reward_mean','config/clip_param','config/lr','episodes_total','hist_stats/episode_reward','hist_stats/episode_lengths']]


# df_3 = all_dfs[3]
# df_time_steps_3 = df_3[['agent_timesteps_total', 'episode_reward_mean','config/clip_param','config/lr','episodes_total','hist_stats/episode_reward','hist_stats/episode_lengths']]


# df_4 = all_dfs[4]
# df_time_steps_4 = df_4[['agent_timesteps_total', 'episode_reward_mean','config/clip_param','config/lr','episodes_total','hist_stats/episode_reward','hist_stats/episode_lengths']]




# df_2 = all_dfs[2]
# df_time_steps_2 = df_2['agent_timesteps_total']
# df_episode_reward_mean_2 = df_2['episode_reward_mean']
# df_clip_param_2 = df_2['config/clip_param']
# df_lr_2 = df_2['config/lr']
# df_hist_reward_2 = df_2['hist_stats/episode_reward']
# df_hist_lengths_2 = df_2['hist_stats/episode_lengths']
# df_episodes_total_2 = df_2['episodes_total']



# df_3 = all_dfs[3]
# df_time_steps_3 = df_3['agent_timesteps_total']
# df_episode_reward_mean_3 = df_3['episode_reward_mean']
# df_clip_param_3 = df_3['config/clip_param']
# df_lr_3 = df_3['config/lr']
# df_hist_reward_3 = df_3['hist_stats/episode_reward']
# df_hist_lengths_3 = df_3['hist_stats/episode_lengths']
# df_episodes_total_3 = df_3['episodes_total']



	
# reward_mean = df_best['episode_reward_mean']
# hist_reward = np.array(df_best['hist_stats/episode_reward'][1])
# hist_len = np.array(df_best['hist_stats/episode_lengths'][1])


# results = pd.DataFrame()
# for i in range(args.num_samples):
#     df = all_dfs[i]
#     df = df[
#         [
#             "timesteps_total",
#             "episodes_total",
#             "episode_reward_mean",
#             #"info/learner/default_policy/cur_kl_coeff",
#         ]
#     ]
#     df["Agent"] = i
#     results = pd.concat([results, df]).reset_index(drop=True)

# if args.save_csv:
#     if not (os.path.exists("data/" + args.dir)):
#         os.makedirs("data/" + args.dir)

#     results.to_csv("data/{}/seed{}.csv".format(args.dir, str(args.seed)))

