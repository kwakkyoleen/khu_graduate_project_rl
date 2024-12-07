import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import asyncio

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


def mix_weights(local_weights, central_weights, alpha=0.5):
    mixed_weights = {}
    for key in local_weights.keys():
        mixed_weights[key] = alpha * local_weights[key] + (1 - alpha) * central_weights[key]
    return mixed_weights


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 128),
                            nn.Tanh(),
                            nn.Linear(128, 128),
                            nn.Tanh(),
                            nn.Linear(128, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 128),
                            nn.Tanh(),
                            nn.Linear(126, 128),
                            nn.Tanh(),
                            nn.Linear(128, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1)
                    )
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class AsyncMultiAgentManager:
    def __init__(self, local_policies, central_policy, device):
        self.local_policies = local_policies
        self.central_policy = central_policy
        self.device = device

    async def compute_action(self, agent_idx, state):
        """
        개별 에이전트의 행동을 비동기적으로 계산하는 코루틴
        """
        with torch.no_grad():
            if agent_idx < len(self.local_policies) : 
                action, logprob, state_value = self.local_policies[agent_idx].act(state)
            else : 
                action, logprob, state_value = self.central_policy.act(state)
        return action, logprob, state_value

    async def get_all_actions(self, states, bundles):
        """
        모든 에이전트의 행동과 관련 데이터를 비동기적으로 계산
        """
        tasks = [
            self.compute_action(i, states[i * bundles : (i + 1) * bundles]) for i in range(len(self.local_policies) + 1)
        ]
        # gather로 모든 코루틴 실행
        results = []
        pre_results = await asyncio.gather(*tasks)
        for a, b, c in pre_results:
            for i in range(bundles):
                results.append((a[i], b[i], c[i]))
        return results
    

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, env_nums, env_bundles, alpha, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.env_nums = env_nums
        self.env_bundles = env_bundles
        self.policy_nums = env_nums // env_bundles - 1
        self.alpha = alpha
        
        self.buffer = [RolloutBuffer() for _ in range(env_nums)]

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.local_policies = [ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device) for _ in range(self.policy_nums)]
        self.manager = AsyncMultiAgentManager(self.local_policies, self.policy, device)
        self.optimizer = [torch.optim.Adam([
                        {'params': self.local_policies[i].actor.parameters(), 'lr': lr_actor},
                        {'params': self.local_policies[i].critic.parameters(), 'lr': lr_critic}
                    ]) for i in range(self.policy_nums)]

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        for local_p in self.local_policies :
            local_p.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
            for local_p in self.local_policies :
                local_p.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = state.clone().detach()
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 이벤트 루프가 있다면 `run_until_complete`를 사용
                    results = loop.run_until_complete(self.manager.get_all_actions(state, self.env_bundles))
                else:
                    # 새로운 이벤트 루프 실행
                    results = asyncio.run(self.manager.get_all_actions(state, self.env_bundles))
                actions = []
                for i, (action, action_logprob, state_val) in enumerate(results):
                    self.buffer[i].states.append(state[i].clone().detach())
                    self.buffer[i].actions.append(action.squeeze(0))
                    self.buffer[i].logprobs.append(action_logprob.squeeze(0))
                    self.buffer[i].state_values.append(state_val.squeeze(0))
                    actions.append(action.squeeze(0).clone().detach())

                action_tensor = torch.stack(actions)

            return action_tensor.cpu().numpy()
            #     for i, net in enumerate(self.local_policies):
            #         action, action_logprob, state_val = net.act(state[i].unsqueeze(0))
            #         self.buffer[i].states.append(state[i])
            #         self.buffer[i].actions.append(action.squeeze(0))
            #         self.buffer[i].logprobs.append(action_logprob.squeeze(0))
            #         self.buffer[i].state_values.append(state_val.squeeze(0))
            #         actions.append(action.squeeze(0))

            #     action_tensor = torch.stack(actions)

            # return action_tensor.cpu().numpy()
        else:
            with torch.no_grad():
                state = state.clone().detach()
                actions = []
                for i, net in enumerate(self.local_policies):
                    action, action_logprob, state_val = net.act(state[i].unsqueeze(0))
                    self.buffer[i].states.append(state[i])
                    self.buffer[i].actions.append(action.squeeze(0))
                    self.buffer[i].logprobs.append(action_logprob.squeeze(0))
                    self.buffer[i].state_values.append(state_val.squeeze(0))
                    actions.append(action.squeeze(0))

                action_tensor = torch.stack(actions)

            return action_tensor.tolist()

    def update(self, env_list):
        # Monte Carlo estimate of returns
        for env_idx in range(self.env_nums):

            if env_idx < self.policy_nums * self.env_bundles:
                env_rewards = self.buffer[env_idx].rewards
                env_states = self.buffer[env_idx].states 
                env_actions = self.buffer[env_idx].actions 
                env_logprobs = self.buffer[env_idx].logprobs 
                env_state_values = self.buffer[env_idx].state_values
                env_is_terminals = self.buffer[env_idx].is_terminals 

                rewards = []
                discounted_reward = 0
                for reward, is_terminal in zip(reversed(env_rewards), reversed(env_is_terminals)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + (self.gamma * discounted_reward)
                    rewards.insert(0, discounted_reward)
                    
                # Normalizing the rewards
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

                # convert list to tensor
                old_states = torch.squeeze(torch.stack(env_states, dim=0)).to(device)
                old_actions = torch.squeeze(torch.stack(env_actions, dim=0)).to(device)
                old_logprobs = torch.squeeze(torch.stack(env_logprobs, dim=0)).to(device)
                old_state_values = torch.squeeze(torch.stack(env_state_values, dim=0)).to(device)

                # calculate advantages
                advantages = rewards.detach() - old_state_values.detach()

                # Optimize policy for K epochs
                for _ in range(self.K_epochs):

                    # Evaluating old actions and values
                    logprobs, state_values, dist_entropy = self.local_policies[env_idx // self.env_bundles].evaluate(old_states, old_actions)

                    # match state_values tensor dimensions with rewards tensor
                    state_values = torch.squeeze(state_values)
                    
                    # Finding the ratio (pi_theta / pi_theta__old)
                    ratios = torch.exp(logprobs - old_logprobs.detach())

                    # Finding Surrogate Loss  
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                    # final loss of clipped objective PPO
                    loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
                    # take gradient step
                    self.optimizer[env_idx // self.env_bundles].zero_grad()
                    loss.mean().backward()
                    self.optimizer[env_idx // self.env_bundles].step()

                if env_idx % self.env_bundles == self.env_bundles - 1 :
                    # 로컬 폴리시에 데이터 가져오기
                    central_state_dict = self.policy.state_dict()

                    local_state_dict = self.local_policies[env_idx // self.env_bundles].state_dict()
                    # 가중치를 섞음
                    mixed_weights = mix_weights(local_state_dict, central_state_dict, alpha=self.alpha)
                    central_state_dict.update(mixed_weights)

                    # policy들 가중치 업데이트
                    self.policy.load_state_dict(central_state_dict)
                    # self.local_policies[env_idx].load_state_dict(self.policy.state_dict())
            self.buffer[env_idx].clear()
                
        # # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # # clear buffer
        # self.buffer.clear()

    def update_central(self):
        central_state_dict = self.policy.state_dict()
        for key in central_state_dict.keys():
        # 모든 로컬 정책의 해당 가중치를 평균 (또는 가중합)
            central_state_dict[key] = self.alpha * torch.mean(
                torch.stack([local_policy.state_dict()[key] for local_policy in self.local_policies]), dim=0
            ) + (1 - self.alpha) * central_state_dict[key]
        
        # 중앙 정책 갱신
        self.policy.load_state_dict(central_state_dict)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 로컬 정책 갱신
        for local_policy in self.local_policies:
            local_policy.load_state_dict(self.policy.state_dict())

        for buf in self.buffer:
            buf.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        for local_p in self.local_policies:
            local_p.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

        
        
       


