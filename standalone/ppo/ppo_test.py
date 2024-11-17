import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Obstacle avoiding task env.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=64, help="Number of environments to simulate."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp
import numpy as np

from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

from mydofbot import MY_UR10E_CFG
from mydofbot.envcfg.obstacle_env import ObstacleEnvCfg

from PPO import PPO
import os
from datetime import datetime

wp.init()

if(torch.cuda.is_available()) : 
    device = torch.device('cuda:0')


class ArmSm:
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)


def main():
    ####### initialize environment hyperparameters ######
    env_name = "Obstacle-direct-v0"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 460                    # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    # parse configuration
    env_cfg: ObstacleEnvCfg = parse_env_cfg(
        "Obstacle-direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make("Obstacle-direct-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 9      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    # initialize a PPO agent
    torch.cuda.empty_cache()
    ppo_agent = PPO(15, 6, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    if os.path.exists(checkpoint_path):
        ppo_agent.load(checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0



    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    print(env.unwrapped.action_space.shape)

    rewards = np.zeros(env.unwrapped.num_envs, dtype=np.float32)
    rewards_latest = np.zeros(env.unwrapped.num_envs, dtype=np.float32)
    total_steps = 0

    ob, r, _, dones, _ = env.step(actions)
    joint = ob["joint"]
    state = joint.clone().detach()
    current_ep_reward = 0
    # dummy_action = torch.tensor([0,-0.6,0,0,0,0], device=device)
    # dummy_action = dummy_action.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode(False):
            # select action with policy
            action = ppo_agent.select_action(state)
            ob, reward, terminated, truncated, _ = env.step(torch.tensor(action, device=device))
            njoint = ob["joint"]
            done = terminated | truncated
            state = njoint.detach().clone()

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward.detach().clone())
            ppo_agent.buffer.is_terminals.append(done.detach().clone())

            time_step += 1
            rewards += reward.cpu().numpy()
            current_ep_reward += reward.cpu().numpy()[0]

            if np.any(done.cpu().numpy()):
                rewards_latest = rewards_latest * (1 - done.cpu().numpy())
                rewards_latest += rewards * done.cpu().numpy()
                rewards = rewards * (1 - done.cpu().numpy())
            
            # # test update PPO agent
            # if time_step % 10 == 0:
            #     print("update..")
            #     ppo_agent.update()

            # value
            if time_step % 50 == 0:
                temp_value = ppo_agent.policy.critic(state)[3]
                temp_action = ppo_agent.policy.actor(state)[3]
                print(f"vel_target : {[round(x, 2) for x in env.temp_target_pos[0].tolist()]}")
                print(f"machine 3 state : {[round(x, 2) for x in state[3].tolist()]}")
                print(f"machine 3 value : {round(temp_value.item(), 2)}, action : {[round(x, 2) for x in temp_action.tolist()]}")
            # update PPO agent
            if time_step % update_timestep == 0:
                print("update..")
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print("step:{} rewards: {}".format(i_episode, rewards_latest))
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if done.cpu().numpy()[0] :
                print_running_reward += current_ep_reward
                print_running_episodes += 1

                log_running_reward += current_ep_reward
                log_running_episodes += 1

                i_episode += 1
                current_ep_reward = 0

    # close the environment
    log_f.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
