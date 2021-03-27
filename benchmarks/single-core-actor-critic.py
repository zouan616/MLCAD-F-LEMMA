import os
import sys
import sim
import heapq
import random
import operator
import itertools
from itertools import count
from collections import namedtuple

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal


import time as python_time
np.random.seed(42)
torch.manual_seed(42)
# torch.set_num_threads(1)

WORKDIR = '/Users/karthik/Desktop/sunday_funday/modified/sniper/benchmarks'
print "RUNNING SINGLE NETWORK MULTI-SCORE ACTOR CRITIC (LOCAL)"
sys.path.append(WORKDIR)
import pacman_power
import single_pacman_network

def select_action(state):
    mu, sigma, state_value = model(state)
    m = Normal(mu, sigma)
    # create a categorical distribution over the list of probabilities of actions
    # m = Categorical(probs)
    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(single_pacman_network.SavedAction(m.log_prob(action), state_value))
    # the action to take (left or right)
    return np.maximum(action.item(), 0.01)




model = single_pacman_network.Policy()

if 'fft_shared_network_model.pth' in os.listdir(WORKDIR):
    print "loading previously saved network"
    model.load_state_dict(torch.load(WORKDIR + '/fft_shared_network_model.pth'))
optimizer = optim.Adam(model.parameters(), lr = 0.001)


# model = single_pacman_network.Policy(input_size, output_size)
# if 'ac_model_pacman.pth' in os.listdir(WORKDIR):
#   print "loading previously saved network"
#   model.load_state_dict(torch.load(WORKDIR + '/ac_model_pacman.pth'))

# optimizer = optim.Adam(model.parameters(), lr=3e-2)
# print sim.config.ncores, "NUMBER OF CORES"

class RLAgent:

  def setup(self, args):
    self.run_RL_agents   = True
    self.reward_function = 0    # reward function to choose 0, 1, or 2
    self.peak_IPC        = 5.0  # Maximum IPC
    self.peak_power      = 12.0 # Maximum Watts
    self.curr_weights    = []

    # setup energystats object: we will get the
    # current dynamic power from this object
    self.energy_stats = pacman_power.EnergyStats()
    sim.util.register(self.energy_stats)
    self.avg_ipcs   = []   # running average of ipc over all cores
    self.avg_mpki   = []   # running average of mpki over all cores
    # updated values to work with McPat
    self.start_Vdd   = 0.96
    self.start_freq = 1600
    self.n_steps    = 0    # n_steps taken
    self.theta      = 0.1  # learning rate for Q-learning
    self.gamma      = 0.9  # future discounted reward rate
    self.epsilon    = 0.1  # exploration probability for epsilon-greedy
    self.eps        = np.finfo(np.float32).eps.item()

    self.available_freqs      = [2000, 1800, 1600, 1400] # actions we can take
    self.VFs                  = [1.2, 1.08, 0.96, 0.84]
    self.mean_vdd             = np.mean(self.VFs)
    self.std_vdd              = np.std(self.VFs)
    self.actions = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


    self.global_budget = 95.0  # total power budget available
    self.core_budgets  = []
    self.PF            = 5.0   # penalty factor
    self.kappa         = 0.9   # transition cost of VF level
    self.curr_Vdds     = []

    self.dvfs_table = pacman_power.build_dvfs_table(int(sim.config.get('power/technology_node')))

    # produces cartesian product given components of state space

    # set up q-table and init frequency of each core.
    for core in range(sim.config.ncores):
      self.curr_weights.append(1)
      self.avg_ipcs.append([])
      self.avg_mpki.append([])
      self.core_budgets.append(0)
      sim.dvfs.set_frequency(core, self.start_freq)



    args = dict(enumerate((args or '').split(':')))
    filename = None
    interval_ns   = long(4000)
    q_interval_ns = long(args.get(0,500000))
    g_interval_ns = q_interval_ns * 15
    self.Q        = int(q_interval_ns / interval_ns)
    self.M        = int(g_interval_ns / interval_ns)


    print "Pacman Interval       :", interval_ns
    print "Q-Learning Interval   :", q_interval_ns
    print "Global Budget Interval:", g_interval_ns

    print("Q:", self.Q)
    print("M:", self.M)

    if filename:
      self.fd = file(os.path.join(sim.config.output_dir, filename), 'w')
      self.isTerminal = False
    else:
      self.fd = sys.stdout
      self.isTerminal = True
    self.sd = sim.util.StatsDelta()
    self.stats = {
      'time': [ self.sd.getter('performance_model', core, 'elapsed_time') for core in range(sim.config.ncores) ],
      'ffwd_time': [ self.sd.getter('fastforward_performance_model', core, 'fastforwarded_time') for core in range(sim.config.ncores) ],
      'instrs': [ self.sd.getter('performance_model', core, 'instruction_count') for core in range(sim.config.ncores) ],
      'coreinstrs': [ self.sd.getter('core', core, 'instructions') for core in range(sim.config.ncores) ],
      'misses' : [self.sd.getter('branch_predictor', core, 'num-incorrect') for core in range(sim.config.ncores)],
    }
    sim.util.Every(interval_ns * sim.util.Time.NS, self.periodic, statsdelta = self.sd, roi_only = True)
    sim.hooks.register(sim.hooks.HOOK_SIM_END, self.finish_episode)

  def get_vdd_from_freq(self, f):
    # Assume self.dvfs_table is sorted from highest frequency to lowest
    for _f, _v in self.dvfs_table:
      if f >= _f:
        return _v
    assert ValueError('Could not find a Vdd for invalid frequency %f' % f)



  def estimate_power(self, core, v, f):
      """
      Function to estimate the power of a core at a particular VF pairs
      inputs:
      core: core to estimate power level
      v   : voltage   level at which we are trying to estimate power
      f   : frequency level at which we are trying to estimate power
      """
      current_power = self.energy_stats.power[('core', core)].d + self.energy_stats.power[('core', core)].s
      curr_f        = sim.dvfs.get_frequency(core)
      curr_v        = self.get_vdd_from_freq(curr_f)
      ratio         = (v*v*f) / (curr_v * curr_v * curr_f) * self.kappa
      return current_power * ratio

  def finish_episode(self):
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    iter_num = 0
    # reversed_rewards = np.copy(model.rewards[::-1])
    # for i in range(0, len(reversed_rewards), sim.config.ncores):
    #     R = sum(reversed_rewards[i:i+2]) + self.gamma * R
    #     returns.insert(0, R)

    for r in model.rewards[::-1]:
        if iter_num % sim.config.ncores == 0:
            R = r + self.gamma * R
        else:
            R = r + R
        # calculate the discounted value
        # R = r + self.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + self.eps)
    print('NUMBER OF REWARDS:', len(model.rewards))
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    # print("policy losses:", policy_losses)
    # print("value losses:", value_losses)
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    # for param in models[core].parameters():
        # param.grad.data.clamp_(-1,1)
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]
    torch.save(model.state_dict(), WORKDIR + '/fft_shared_network_model.pth')

  def periodic(self, time, time_delta):
    self.n_steps += 1
    ## COURSE GRAIN GLOBAL BUDGET ALLOCATOR
    if (self.n_steps % self.M == 0) and self.run_RL_agents:
        current_real_time = python_time.time()
        current_global_budget = self.global_budget
        core_ipc = []
        print "Running global budget allocator"
        for core in range(sim.config.ncores):
            self.core_budgets[core] = self.estimate_power(core, min(self.VFs), min(self.actions))
            current_global_budget = current_global_budget - self.core_budgets[core]
            cycles = self.stats['time'][core].delta * sim.dvfs.get_frequency(core) / 1e9 # convert fs to cycles
            instrs = self.stats['coreinstrs'][core].delta
            ipc    = instrs / (cycles or 1)
            heapq.heappush(core_ipc, (-1*ipc, core)) # we use -1 * ipc to make the min a max
        print(core_ipc)
        while (current_global_budget > 0) and (len(core_ipc) > 0):
            ipc, core = heapq.heappop(core_ipc)
            ipc = -1 * ipc # revert ipc back to actual value
            # print "GB - IPC, CORE:", ipc, core
            delta = self.estimate_power(core, max(self.VFs), max(self.actions)) - self.core_budgets[core]
            print "GB - IPC, CORE, DELTA:", ipc, core, delta
            if delta <= current_global_budget:
                self.core_budgets[core] = self.core_budgets[core] + delta
                current_global_budget   = current_global_budget - delta
            else:
                self.core_budgets[core] = self.core_budgets[core] + current_global_budget
                current_global_budget   = 0
        print("GLOBAL BUDGET TIME, BUDGETS:", python_time.time() - current_real_time, self.core_budgets)
        print("")


    if self.isTerminal:
      self.fd.write('[IPC MPKI] ')
    self.fd.write('%u ns\n' % (time / 1e6)) # Time in ns
    qq = 0

    ## FINE GRAIN DISTRIBUTED Q-LEARNING
    # if self.n_steps == 0 or self.n_steps % self.Q == 0:
    new_weights = []
    for core in range(sim.config.ncores):
      # detailed-only IPC
      cycles = (self.stats['time'][core].delta - self.stats['ffwd_time'][core].delta) * sim.dvfs.get_frequency(core) / 1e9 # convert fs to cycles
      instrs = self.stats['instrs'][core].delta
      ipc = instrs / (cycles or 1) # Avoid division by zero
      #self.fd.write(' %.3f' % ipc)

      # include fast-forward IPCs
      cycles = self.stats['time'][core].delta * sim.dvfs.get_frequency(core) / 1e9 # convert fs to cycles
      instrs = self.stats['coreinstrs'][core].delta
    #   maybe_mpki = self.stats['check_mpki'][core].delta / instrs
    #   print(dir(self.stats['check_mpki'][core]))
      if instrs != 0:
          mpki = self.stats['misses'][core].delta / instrs * 1000
      else:
          mpki = 0
      ipc = instrs / (cycles or 1)

      self.fd.write("Core " + str(core))
      self.fd.write(' %.3f' % ipc)
      self.fd.write(' %.3f' % mpki)
      self.fd.write('\n')


      # update global and average ipc and mpki values
      self.avg_ipcs[core].append(ipc)
      self.avg_mpki[core].append(mpki)
      # self.avg_ipcs[core]  = ((self.n_steps-1) * self.avg_ipcs[core] + ipc) / (self.n_steps)
      # self.avg_mpki[core]  = ((self.n_steps-1) * self.avg_mpki[core] + mpki) / (self.n_steps)

      if (self.n_steps % self.Q == 0) and self.run_RL_agents:
        print "Running Actor-Critic"
        curr_power = self.energy_stats.power[('core', core)].d + self.energy_stats.power[('core', core)].s
        normalized_ipc   = (ipc - np.mean(self.avg_ipcs[core])) / (np.std(self.avg_ipcs[core]) or 1)
        normalized_mpki  = (mpki - np.mean(self.avg_mpki[core])) / (np.std(self.avg_mpki[core]) or 1)
        normalized_vdd   = (self.get_vdd_from_freq(sim.dvfs.get_frequency(core)) - self.mean_vdd) / self.std_vdd
        normalized_power = curr_power / self.peak_power

        state = torch.tensor([normalized_ipc, normalized_mpki, normalized_vdd, normalized_power])
        reward = ipc
        reward = reward - self.PF * abs(curr_power - self.core_budgets[core])
        model.rewards.append(reward / sim.config.ncores)
        action = select_action(state)
        action = action if action > 0 else 0.0
        new_weights.append(action)
        print("NEW WEIGHTS:", new_weights)


    if new_weights:
      for i in range(len(new_weights)):
        self.curr_weights[i] = new_weights[i]
    print("WEIGHTS:", self.curr_weights)
    sum_power = 0
    for core in range(sim.config.ncores):
      sum_power = sum_power + self.energy_stats.power[('core', core)].d + self.energy_stats.power[('core', core)].s

    sum_weights = sum(self.curr_weights) + 0.0
    normalized_weights = []
    for i in range(len(self.curr_weights)):
      # normalized_weights.append((self.curr_weights[i] - min_weight) / ((max_weight - min_weight) or 1.))
      normalized_weights.append(self.curr_weights[i] / (sum_weights + 0.0))
      # new_weights[i] /= (sum_weights + 0.0)
    print 'NORMLIZED WIEGHTS', normalized_weights
    final_powers = []
    for core in range(sim.config.ncores):
      curr_core_power = self.energy_stats.power[('core', core)].d + self.energy_stats.power[('core', core)].s
      final_powers.append(normalized_weights[core] * sum_power)
      curr_freq = sim.dvfs.get_frequency(core)
      print "Weighted Power.{}:".format(core), final_powers[core]
      print "Actual Power.{}  :".format(core), curr_core_power
      if final_powers[core] > curr_core_power:
        if curr_freq < max(self.available_freqs):
          curr_idx = self.available_freqs.index(curr_freq)
          sim.dvfs.set_frequency(core, self.available_freqs[curr_idx - 1])

        # increase the dvfs pair
      elif final_powers[core] < curr_core_power:
        if curr_freq > min(self.available_freqs):
          curr_idx = self.available_freqs.index(curr_freq)
          sim.dvfs.set_frequency(core, self.available_freqs[curr_idx + 1])

    self.fd.write('\n')


sim.util.register(RLAgent())
