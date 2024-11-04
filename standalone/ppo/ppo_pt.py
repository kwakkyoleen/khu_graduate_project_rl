from PPO import PPO
import os

####### initialize environment hyperparameters ######
env_name = "Obstacle-direct-v0"

has_continuous_action_space = True  # continuous action space; else discrete

max_ep_len = 460                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)          # save model frequency (in num timesteps)

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

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path1 = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
checkpoint_path2 = directory + "PPO_{}_{}_{}_cp.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path1)
#####################################################

# initialize a PPO agent
ppo_agent = PPO(37, 6, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

ppo_agent.load(checkpoint_path1)


for name, param in ppo_agent.policy.state_dict().items():
    print(f"Layer: {name} | Weights: {param}")
    break

ppo_agent.load(checkpoint_path2)
for name, param in ppo_agent.policy.state_dict().items():
    print(f"Layer: {name} | Weights: {param}")
    break
