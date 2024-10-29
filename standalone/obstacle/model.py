import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
from replay_buffer import PrioritizedReplayBuffer
from pathlib import Path


# state -> image, angle + tcp, target으로 나누어 받을 예정
class Actor(nn.Module):
    def __init__(
        self, tcp_dim, action_dim=1, log_std_min=-20, log_std_max=2
    ):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 로봇 상태 처리부
        self.tfc1 = nn.Linear(tcp_dim, 512)
        self.tfc2 = nn.Linear(512, 256)
        self.tfc3 = nn.Linear(256, 256)

        self.fcmean = nn.Linear(256, action_dim)
        self.fccov = nn.Linear(256, action_dim)

    def forward(self, tcp):
        x2 = F.leaky_relu(self.tfc1(tcp))
        x2 = F.leaky_relu(self.tfc2(x2))
        x2 = F.leaky_relu(self.tfc3(x2))

        # get mean
        mu = self.fcmean(x2).tanh()

        # get std
        log_std = self.fccov(x2).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        std = torch.exp(log_std)

        # sample actions
        dist = Normal(mu, std)
        z = dist.rsample()

        # normalize action and log_prob
        # see appendix C of [2]
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, mu, log_prob


class CriticQ(nn.Module):
    def __init__(self, tcp_dim, action_dim=1):
        super(CriticQ, self).__init__()
        self.fc1 = nn.Linear(action_dim + tcp_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, tcp, action):
        combined = torch.cat((tcp, action), dim=1)
        combined = F.leaky_relu(self.fc1(combined))
        combined = F.leaky_relu(self.fc2(combined))
        combined = F.leaky_relu(self.fc3(combined))
        combined = self.fc4(combined)
        return combined


class CriticV(nn.Module):
    def __init__(self, tcp_dim):
        super(CriticV, self).__init__()
        self.fc1 = nn.Linear(tcp_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, tcp):
        combined = F.leaky_relu(self.fc1(tcp))
        combined = F.leaky_relu(self.fc2(combined))
        combined = F.leaky_relu(self.fc3(combined))
        combined = self.fc4(combined)
        return combined


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.ns = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, ns, done):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.ns[ind]),
            torch.FloatTensor(self.done[ind]),
        )


class SAC:
    def __init__(self, tcp_dim, action_dim):
        self.lr = 0.01
        self.lrq = 0.01
        self.lra = 0.0003
        self.gamma = 0.99
        self.batch_size = 1024
        self.buffer_size = 3000000
        self.warmup_steps = 100
        self.tau = 5e-3
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        self.actor_range = (-2, 2)
        self.total_steps = 0
        self.policy_update_freq = 2
        # self.alpha = 0.2

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.target_entropy = -np.prod((action_dim,)).item()

        self.actor = Actor(tcp_dim, action_dim).to(self.device)
        self.criticQ1 = CriticQ(tcp_dim, action_dim).to(self.device)
        self.criticQ2 = CriticQ(tcp_dim, action_dim).to(self.device)
        self.criticV = CriticV(tcp_dim).to(self.device)
        # self.criticQ_target1 = CriticQ(state_dim, action_dim).to(self.device)
        # self.criticQ_target2 = CriticQ(state_dim, action_dim).to(self.device)
        self.criticV_target = CriticV(tcp_dim).to(self.device)
        # self.criticQ_target1.load_state_dict(self.criticQ1.state_dict())
        # self.criticQ_target2.load_state_dict(self.criticQ2.state_dict())
        self.criticV_target.load_state_dict(self.criticV.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.lra)
        self.criticQ_optimizer1 = torch.optim.Adam(self.criticQ1.parameters(), self.lrq)
        self.criticQ_optimizer2 = torch.optim.Adam(self.criticQ2.parameters(), self.lrq)
        self.criticV_optimizer = torch.optim.Adam(self.criticV.parameters(), self.lr)
        self.buffer = PrioritizedReplayBuffer(batch_size=self.batch_size, capacity=self.buffer_size)

    @property
    def alpha(self):
        # return self.log_alpha.exp()
        return 0.005

    def select_action(self, joint, target, training=True):
        tcp = torch.cat((joint, target), dim=1).clone().detach().to(torch.float32).to(self.device)
        # img = torch.tensor(img.clone(), dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        # tcp = torch.tensor(torch.cat((joint, target), dim=1).clone(), dtype=torch.float32, device=self.device)
        # img = torch.FloatTensor(img).to(self.device).permute(0, 3, 1, 2)
        # tcp = torch.FloatTensor(np.concatenate((joint, target), axis=1)).to(self.device)
        action, mu, log_prob = self.actor(tcp)
        # action = action.detach().cpu().numpy()[0]
        # mu = mu.detach().cpu().numpy()[0]
        # log_prob = log_prob.detach().cpu().numpy()[0]
        if training:
            return action
        else:
            return mu.detach()

    def select_action_log(self, tcp, training=True):
        # x = torch.tensor(x, dtype=torch.float32).to(self.device).unsqueeze(0)
        action, mu, log_prob = self.actor(tcp)
        action = action.detach().cpu().numpy()[0]
        mu = mu.detach().cpu().numpy()[0]
        log_prob = log_prob.detach().cpu().numpy()[0]
        if training:
            return action
        else:
            return mu

    def update_model(self):
        # 쓸데없는거 지우기?
        # torch.cuda.empty_cache()
        # 에피소드 뽑아오기
        # s, a, r, ns, done = map(
        #     lambda x: torch.FloatTensor(x).to(self.device),
        #     self.buffer.sample(self.batch_size),
        # )
        _, transition, _ = self.buffer.sample()
        done = torch.tensor(transition["done"], dtype=torch.float32, device=self.device)
        tcp = torch.tensor(np.concatenate((transition["joint"], transition["target"]), axis=1), dtype=torch.float32, device=self.device)
        a = torch.tensor(transition["action"], dtype=torch.float32, device=self.device)
        r = torch.tensor(transition["reward"], dtype=torch.float32, device=self.device)
        ntcp = torch.tensor(np.concatenate((transition["njoint"], transition["ntarget"]), axis=1), dtype=torch.float32, device=self.device)
        
        # td target 구하기
        sa, smu, sa_log = self.actor(tcp)
        alpha = self.log_alpha.exp()

        # q function loss
        r = r.unsqueeze(1)
        mask = 1 - done
        mask = mask.unsqueeze(1)
        q_1_pred = self.criticQ1(tcp, a)
        q_2_pred = self.criticQ2(tcp, a)
        v_target = self.criticV_target(ntcp)
        q_target = r + self.gamma * v_target * mask
        # print("Shape of mask:", mask.shape) 
        # print("Shape of vtarget:", v_target.shape)
        # print("Shape of qtarget:", q_target.shape)
        # print("Shape of r:", r.shape)
        # print("Shape of q_pred:", q_1_pred.shape)
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        # v function loss
        v_pred = self.criticV(tcp)
        q_pred = torch.min(self.criticQ1(tcp, sa), self.criticQ2(tcp, sa))
        v_target = q_pred - alpha * sa_log
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_steps % self.policy_update_freq == 0:
            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * sa_log - advantage).mean()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.requires_grad_(True)
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update (vf)
            # 타겟 업데이트
            for target_param, param in zip(
                self.criticV_target.parameters(), self.criticV.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        else:
            actor_loss = torch.zeros(1)

        # criticV 학습
        self.criticV_optimizer.zero_grad()
        vf_loss.requires_grad_(True)
        vf_loss.backward()
        self.criticV_optimizer.step()

        # criticQ 학습
        self.criticQ_optimizer1.zero_grad()
        qf_1_loss.requires_grad_(True)
        qf_1_loss.backward()
        self.criticQ_optimizer1.step()

        self.criticQ_optimizer2.zero_grad()
        qf_2_loss.requires_grad_(True)
        qf_2_loss.backward()
        self.criticQ_optimizer2.step()

        # 알파 값 학습

        alpha_loss = (
            -self.log_alpha.exp() * (sa_log + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.requires_grad_(True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def load_param(self, apath, vpath, q1path, q2path, bufferpath):
        success_flag = True
        if os.path.isfile(apath):
            self.actor.load_state_dict(torch.load(apath))
        else:
            success_flag = False
        if os.path.isfile(vpath):
            self.criticV.load_state_dict(torch.load(vpath))
            self.criticV_target.load_state_dict(torch.load(vpath))
        else:
            success_flag = False
        if os.path.isfile(q1path):
            self.criticQ1.load_state_dict(torch.load(q1path))

        else:
            success_flag = False
        if os.path.isfile(q2path):
            self.criticQ2.load_state_dict(torch.load(q2path))

        else:
            success_flag = False
        # self.buffer.load(Path('.'), bufferpath)
        return success_flag

    def save_param(self, apath, vpath, q1path, q2path, bufferpath):
        torch.save(self.actor.state_dict(), apath)
        torch.save(self.criticV.state_dict(), vpath)
        torch.save(self.criticQ1.state_dict(), q1path)
        torch.save(self.criticQ2.state_dict(), q2path)
        # self.buffer.save(Path('.'), bufferpath)

    # transit은 dict 형태 (transit = {state:np.arr, action:np,arr...})
    def process(self, transit):
        self.total_steps += 1
        self.buffer.add(transit)

        if self.total_steps > self.warmup_steps:
            self.update_model()


if __name__ == "__main__":
    # input_tensor = input_tensor.permute(0, 3, 1, 2)
    sac = SAC(9, 6)
    batch = 8
    joint = np.random.rand(batch, 6)
    target = np.random.rand(batch, 3)
    sac.select_action(joint, target)
