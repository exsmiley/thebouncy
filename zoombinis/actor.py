"""
Inspired by https://github.com/MorvanZhou/pytorch-A3C
"""

import torch
import torch.nn as nn
from actor_utils import v_wrap, set_init, push_and_pull, record, SharedAdam
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
from zoombinis import *
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
MAX_EP = 5000

env = GameEnv()
N_S = env.state_size
N_A = env.action_size


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 1000)
        self.pi2 = nn.Linear(1000, a_dim)
        self.v1 = nn.Linear(s_dim, 1000)
        self.v2 = nn.Linear(1000, 1)
        # set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.relu(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s, verbose=False):
        self.eval()
        logits, _ = self.forward(s)

        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        if verbose:
            print('probs', prob.numpy()[0])
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach()
        a_loss = -exp_v
        # print('Loss', c_loss, a_loss)
        total_loss = (c_loss + a_loss).mean()
        return total_loss

    def load(self, name='models/actor'):
        self.load_state_dict(torch.load(name))
        print('Loaded model from {}!'.format(name))

    def save(self, name='models/actor'):
        torch.save(self.state_dict(), name)
        print('Saved model to {}!'.format(name))

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = GameEnv()

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                already_made_move = self.env.check_already_made(a)
                s_, r, done = self.env.step(a)

                # TODO update rewards
                # if already_made_move:
                #     continue
                #     r = -1
                #     print('BAD')
                # print(done, r, a)
                # if done: r = 1;print('done!')
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)

class Tester(object):

    def __init__(self):
        self.net = Net(N_S, N_A)
        self.net.load()
        self.env = GameEnv()

    def run(self):
        s = self.env.reset()
        done = False
        reward = 0
        num_steps = 0

        print('Playing game!')
        print(self.env.game)

        while not done:
            a = self.net.choose_action(v_wrap(s[None, :]), verbose=True)
            already_made_move = self.env.check_already_made(a)
            if already_made_move:
                continue
            print('Action', a)
            s, r, done = self.env.step(a, verbose=True)
            reward += r
            num_steps += 1

        print('Steps {} | Reward {}'.format(num_steps, reward))

        # for z in self.env.game.zoombinis:
        #     print(z.get_agent_vector())


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-3)      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    NUM_WORKERS = 1#mp.cpu_count()
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(NUM_WORKERS)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    gnet.save()

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

    Tester().run()
