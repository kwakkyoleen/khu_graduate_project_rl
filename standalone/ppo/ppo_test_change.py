import argparse

from omni.isaac.lab.app import AppLauncher
import copy

# add argparse arguments
parser = argparse.ArgumentParser(description="Obstacle avoiding task env.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=128, help="Number of environments to simulate."
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

from PPO_change import PPO
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

    max_ep_len = 480                    # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    env_bundles = 16
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    central_update_idx = 0
    central_update_triggers = [0.1, 0.07, 0.05, 0.03]
    central_update_multiples = [2, 3, 5, 8, 12]
    K_epochs = 10               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    alpha = 0.05

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

    central_idx = env.unwrapped.num_envs - env_bundles

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
    acc_f_name = log_dir + '/PPO_' + env_name + "_acc_" + str(run_num) + ".csv"
    act_f_name = log_dir + '/PPO_' + env_name + "_act_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 34     #### change this to prevent overwriting weights in same env_name folder

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
    ppo_agent = PPO(15, 6, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, env.unwrapped.num_envs, env_bundles, alpha, action_std)
    if os.path.exists(checkpoint_path):
        print("load exist pth")
        ppo_agent.load(checkpoint_path)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    # acc_f = open(acc_f_name, "w+")
    # act_f = open(act_f_name, "w+")
    log_f.write('episode,timestep,reward,precision,actvar,reach_step,reach_ratio,grade\n')
    # acc_f.write('timestep,acc\n')
    # act_f.write('timestep,action\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    min_acc = 100



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
    now_bundle = 0
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
            for env_idx in range(env.unwrapped.num_envs):
                ppo_agent.buffer[env_idx].rewards.append(reward[env_idx].detach().clone())
                ppo_agent.buffer[env_idx].is_terminals.append(done[env_idx].detach().clone())

            time_step += 1
            rewards += reward.cpu().numpy()
            current_ep_reward += np.mean(reward.cpu().numpy()[central_idx:])

            if min_acc > round(env.unwrapped.target_distance[0].item(), 4):
                min_acc = round(env.unwrapped.target_distance[0].item(), 4)

            #logging
            # temp_action = ppo_agent.policy.actor(state)[3]
            # acc_f.write('{},{}\n'.format(time_step, round(env.unwrapped.target_distance[3].item(), 4)))
            # acc_f.flush()
            # act_f.write('{},{}\n'.format(time_step, [round(x, 4) for x in temp_action.tolist()]))
            # act_f.flush()

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
                temp_value = ppo_agent.local_policies[0].critic(state[3].unsqueeze(0)).squeeze(0)
                temp_action = ppo_agent.local_policies[0].actor(state[3].unsqueeze(0)).squeeze(0)
                # print(f"vel_target : {[round(x, 2) for x in env.temp_target_pos[0].tolist()]}")
                print(f"grade : {env.unwrapped.grade}, precision : {env.unwrapped.precision}")
                print(f"machine 3 state : {[round(x, 4) for x in state[3, 0:3].tolist()]}, acc : {round(env.unwrapped.target_distance[0].item(),4)}")
                print(f"machine 3 value : {round(temp_value.item(), 2)}, action : {[round(x, 2) for x in temp_action.tolist()]}")

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                actions = copy.deepcopy(ppo_agent.buffer[central_idx].actions)
                stacked_tensor = torch.stack(actions)
                variance_dim0 = torch.var(stacked_tensor, dim=0)
                var_mean = torch.mean(variance_dim0)

                avg_reach_time = torch.mean(env.unwrapped.min_reach_time_back.clone().float()[central_idx:]).item()
                min_target_distance = env.unwrapped.min_target_distance_back.clone()[central_idx:]
                avg_precision = torch.mean(min_target_distance.clone().float()).item()
                reach_ratio = (torch.sum(min_target_distance < 0.01) / min_target_distance.numel()).item()
                grade = env.unwrapped.grade

                log_f.write('{},{},{},{},{},{},{},{}\n'.format(i_episode, time_step, log_avg_reward, avg_precision, var_mean, avg_reach_time, reach_ratio, grade))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0
                min_acc = 100

            # update PPO agent
            if time_step % update_timestep == 0:
                print("update..")
                bundle_list = [i * env_bundles + now_bundle for i in range(env.unwrapped.num_envs // env_bundles)]
                ppo_agent.update(bundle_list)
                now_bundle += 1
                now_bundle = now_bundle % env_bundles

            # update central
            if time_step % (update_timestep * central_update_multiples[central_update_idx]) == 0:
                if central_update_idx < len(central_update_triggers) and avg_precision < central_update_triggers[central_update_idx]:
                    central_update_idx += 1
                
                print("central update..")
                ppo_agent.update_central()

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

            if done.cpu().numpy()[central_idx] :
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
