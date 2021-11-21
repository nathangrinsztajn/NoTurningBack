import gym
import time
import os
import copy
import argparse
import numpy as np
import json

from stable_baselines3_copy import DQN, PPO
from stable_baselines3_copy.common.monitor import Monitor
from stable_baselines3_copy.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3_copy.common.vec_env.vec_intrisinc_reward import VecIntrinsic
from stable_baselines3_copy.common.callbacks import CallbackList
from stable_baselines3_copy.common.vec_env import SubprocVecEnv
import torch
from torch.nn import Linear

from gym_turf import TurfEnv

from model import ExtractorGrassland

# Training settings
parser = argparse.ArgumentParser()
# Training parameter
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--time_steps', type=int, default=5000000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--ent_coef', type=float, default=0.05,
                    help='PPO entropy coef')
parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument("--log_directory", default="turfRAE",
                    help="The log directory")
parser.add_argument('--save_models', action='store_true', default=False,
                    help='saving the models at the end of the training.')
parser.add_argument('--n_parallel', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.7)
parser.add_argument('--wir', type=float, default=0.1)
parser.add_argument('--offline', action='store_true', default=False)
parser.add_argument('--reward_free', action='store_true', default=False)

parser.add_argument('--train_freq', type=int, default=500)
parser.add_argument('--gradient_step', type=int, default=10)
parser.add_argument('--learning_start', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--buffer_size', type=int, default=2000)
parser.add_argument('--d_min', type=int, default=0)
parser.add_argument('--d_max', type=int, default=100)
parser.add_argument('--alg', type=str, default='PPO')

parser.add_argument('--step_penalty', type=float, default=0)

args = parser.parse_args()

if args.debug:
    print(args)

use_gpu = torch.cuda.is_available() and not args.no_cuda
threshold = args.threshold
wir = args.wir

seed = args.seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

log_dir = args.log_directory
os.makedirs(log_dir, exist_ok=True)


with open(os.path.join(log_dir, 'config'), 'w') as f:
    json.dump(vars(args), f, indent=4)

func = lambda x: (x > threshold) * (x - threshold)

# Logs will be saved in log_dir/monitor.csv
if __name__ == '__main__':

    env = TurfEnv(step_penalty=args.step_penalty)
    env = Monitor(env, os.path.join(log_dir, 'exp'), info_keywords=('ruined grasses',))
    env.seed(seed)

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_dir,
                clip_range_vf=None,
                ent_coef=args.ent_coef)

    head = Linear(2 * 64, 1)
    extractor = ExtractorGrassland(model.env.observation_space, features_dim=64)
    if use_gpu:
        head = head.cuda()
        extractor = extractor.cuda()
    head.train()
    extractor.train()

    model.env = VecIntrinsic(model.env, feature_extractor=extractor, head=head,
                             weight_intrinsic_reward=wir,
                             func=func,
                             train_freq=args.train_freq,
                             gradient_step=args.gradient_step,
                             learning_start=args.learning_start,
                             batch_size=args.batch_size,
                             buffer_size=args.buffer_size,
                             d_min=args.d_min,
                             d_max=args.d_max,
                             reward_free=args.reward_free,
                             save_path=log_dir,
                             )

    model.learn(total_timesteps=args.time_steps)

    if args.save_models:
        model.save(os.path.join(log_dir, 'model.pt'))
