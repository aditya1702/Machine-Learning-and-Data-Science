import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

# hyper parameters
EPISODES = 2000  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.75  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 164  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

env = gym.make('CartPole-v0').unwrapped

model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
ed = []

# def plot_durations(d):
#     plt.figure(2)
#     plt.clf()
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(d)
#
#     plt.savefig('test2.png')

def select_action(state, train=True):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if train:
        if sample > eps_threshold:
            return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(2)]])
    else:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)

def run_episode(episode, env):
    state = env.reset()
    steps = 0
    while True:
        # env.render()
        action = select_action(FloatTensor([state]))
        next_state, reward, done, _ = env.step(action[0, 0])

        # negative reward when attempt ends
        if done:
            if steps < 30:
                reward -= 10
            else:
                reward = -1
        if steps > 100:
            reward += 1
        if steps > 200:
            reward += 1
        if steps > 300:
            reward += 1

        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        learn()

        state = next_state
        steps += 1

        if done or steps >= 1000:
            ed.append(steps)
            print("[Episode {:>5}]  steps: {:>5}".format(episode, steps))
            if sum(ed[-10:])/10 > 800:
                return True
            break
    return False

def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def botPlay():
    state = env.reset()
    steps = 0
    frames = []
    while True:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action = select_action(FloatTensor([state]))
        next_state, reward, done, _ = env.step(action[0, 0])

        state = next_state
        steps += 1

        if done or steps >= 1000:
            break

    clip = ImageSequenceClip(frames, fps=20)
    clip.write_gif('test2.gif', fps=20)

for e in range(EPISODES):
    complete = run_episode(e, env)

    if complete:
        print('complete...!')
        break

# plot_durations(ed)
# botPlay()