import os
import sys
import sim
import json
import heapq
import random
import operator
import itertools
import time as pytime
from itertools import count
from collections import namedtuple

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


import time as python_time
np.random.seed(42)
torch.manual_seed(42)


WORKDIR               = os.environ['BENCHMARKS_ROOT'] #current directory
CURR_BENCHMARK        = os.environ['CURR_BENCHMARK']
REL_SAMPLED_DATA_PATH = 'sample-data-' + CURR_BENCHMARK + '.pt'
ABS_SAMPLED_DATA_PATH = WORKDIR + '/' + REL_SAMPLED_DATA_PATH
sys.path.append(WORKDIR)




import pacman_power_checker_2 as pacman_power_checker
import network_categorical_pacman as single_pacman_network
actions = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def select_action(core, state):
    action_prob, state_value = models[core](state)
    m = Categorical(action_prob)
    # create a Normal distribution based on output of neural network


    action = m.sample()

    models[core].saved_actions.append(single_pacman_network.SavedAction(m.log_prob(action), state_value))

    # clip so that the output of the sample is at least a small positive value.
    return actions[action.item()]


models     = []
optimizers = []
model_name = 'categorical_pacman_{}.pt'
for core in range(sim.config.ncores):
    print 'Creating an actor-critic model for core number: ' + str(core)
    models.append(single_pacman_network.Policy())
    if model_name.format(core) in os.listdir(WORKDIR):
        print 'Loading previously trained model'
        models[core].load_state_dict(torch.load(WORKDIR + '/' + model_name.format(core)))
    optimizers.append(optim.Adam(models[core].parameters(), lr = 0.001))



class RLAgent:

  def setup(self, args):
      """
      Init function (Sniper uses setup() instead of __init__())
      See comments before each codeblock.
      """
      # setup parameters for each core.
      self.frequencies  = [2000, 1800, 1600, 1400]
      self.voltages     = [1.20, 1.08, 0.96, 0.84]
      self.mean_vdd     = np.mean(self.voltages)
      self.std_vdd      = np.std(self.voltages)
      self.min_vdd      = np.min(self.voltages)
      self.max_vdd      = np.max(self.voltages)
      self.min_freq     = np.min(self.frequencies)
      self.max_freq     = np.max(self.frequencies)

      self.prev_data    = torch.load(REL_SAMPLED_DATA_PATH)
      self.freq_to_volt = {}
      for i in range(len(self.frequencies)):
          self.freq_to_volt[self.frequencies[i]] = self.voltages[i]


      # initialize McPAT.
      self.energy_stats = pacman_power_checker.EnergyStats()
      self.dvfs_table   = pacman_power_checker.build_dvfs_table(int(sim.config.get('power/technology_node')))
      sim.util.register(self.energy_stats)


      # setup properties for pacman-like infrastructure.
      self.curr_weights  = [] # weight for each core
      self.core_budgets  = [] # budget for each core
      self.n_steps       = -1 # simulation steps
      self.eps           = np.finfo(np.float32).eps.item() # for numerical stability
      self.global_budget = 95.0 # total power budget available
      self.PF            = 5    # penalty factor in reward function
      self.kappa         = 0.9  # transition cost from moving VF levels

      # unsure if we need these
      self.run_RL_agents = True # run learning based method
      self.gamma         = 0.95S

      # setup initial weights, budget, and set frequency of each core.
      # 3 -> lowest VF  (1400 MHz, 0.84 Volts)
      # 0 -> highest VF (2000 MHz, 1.20 Volts)
      self.start_idx   = 0
      self.start_freq  = self.frequencies[self.start_idx]
      self.start_vdd   = self.voltages[self.start_idx]
      for core in range(sim.config.ncores):
          self.curr_weights.append(1)
          self.core_budgets.append(self.global_budget / sim.config.ncores)
          sim.dvfs.set_frequency(core, self.start_freq)

      # extract time intevals for pacman, RL, Global
      args          = dict(enumerate((args or '').split(':')))
      filename      = None
      interval_ns   = long(4000)
      q_interval_ns = long(args.get(0,500000))
      g_interval_ns = q_interval_ns * 15
      self.Q        = int(q_interval_ns / interval_ns)
      self.M        = int(g_interval_ns / interval_ns)
      self.K        = int(25 * self.Q) #number of iterations before performing backprop


      print "Pacman Interval       :", interval_ns, 1
      print "Q-Learning Interval   :", q_interval_ns, self.Q
      print "Global Budget Interval:", g_interval_ns, self.M

      # setup debugging
      if filename:
          self.fd         = file(os.path.join(sim.config.output_dir, filename), 'w')
          self.isTerminal = False
      else:
          self.fd = sys.stdout
          self.isTerminal = True

      # necessary system statistics for our algorithm to run
      self.sd = sim.util.StatsDelta()
      self.stats = {
                    'time'       : [ self.sd.getter('performance_model', core, 'elapsed_time') for core in range(sim.config.ncores) ],
                    'ffwd_time'  : [ self.sd.getter('fastforward_performance_model', core, 'fastforwarded_time') for core in range(sim.config.ncores) ],
                    'instrs'     : [ self.sd.getter('performance_model', core, 'instruction_count') for core in range(sim.config.ncores) ],
                    'coreinstrs' : [ self.sd.getter('core', core, 'instructions') for core in range(sim.config.ncores) ],
                    'misses'     : [self.sd.getter('branch_predictor', core, 'num-incorrect') for core in range(sim.config.ncores)],
                    'l2stores'     : [self.sd.getter('L2', core, 'stores') for core in range(sim.config.ncores)],
                    'l2storemiss'  : [self.sd.getter('L2', core, 'store-misses') for core in range(sim.config.ncores)],
                    'l2loads'     : [self.sd.getter('L2', core, 'loads') for core in range(sim.config.ncores)],
                    'l2loadmiss'  : [self.sd.getter('L2', core, 'load-misses') for core in range(sim.config.ncores)],
                    'l3loads'          : [self.sd.getter('L3', core, 'loads') for core in range(sim.config.ncores)],
                    'l3loadmisses'     : [self.sd.getter('L3', core, 'load-misses') for core in range(sim.config.ncores)],
                    'l3stores'     : [self.sd.getter('L3', core, 'stores') for core in range(sim.config.ncores)],
                    'l3storemisses'     : [self.sd.getter('L3', core, 'store-misses') for core in range(sim.config.ncores)],
                    'memstores'     : [self.sd.getter('dram', 0, 'writes') ],
                    'memloads'      : [self.sd.getter('dram', 0, 'reads')],
                    'l1-dstores' : [self.sd.getter('L1-D', core, 'stores') for core in range(sim.config.ncores)],
                    'l1-dstoremisses' : [self.sd.getter('L1-D', core, 'store-misses') for core in range(sim.config.ncores)],
                    'l1-dloads'  : [self.sd.getter('L1-D', core, 'loads') for core in range(sim.config.ncores)],
                    'l1-dloadmisses' : [self.sd.getter('L1-D', core, 'load-misses') for core in range(sim.config.ncores)],
                    'idle-time'  : [ self.sd.getter('performance_model', core, 'idle_elapsed_time') for core in range(sim.config.ncores)],
                    'dtlb-accesses' : [ self.sd.getter('dtlb', core, 'access') for core in range(sim.config.ncores) ],
                    'dtlb-misses'   : [ self.sd.getter('dtlb', core, 'miss') for core in range(sim.config.ncores) ]
                    }

      # define periodic function call based on Pacman interval
      sim.util.Every(interval_ns * sim.util.Time.NS, self.periodic, statsdelta = self.sd, roi_only = True)
      sim.hooks.register(sim.hooks.HOOK_SIM_END, self.finish_episode)




  def get_vdd_from_freq(self, f):
      """
      Function to return vdd from frequency value.
      """
      try:
          return self.freq_to_volt[f]
      except:
          assert ValueError('Could not find a Vdd for invalid frequency %f' % f)



  def estimate_power(self, core, v, f):
      """
      From Profit: Priority and Power/Performance
           Optimization for Many-Core Systems
           Equation 3.
      Function to estimate the power of a core at a particular VF pairs
      inputs:
      core: core to estimate power level
      v   : voltage   level at which we are trying to estimate power
      f   : frequency level at which we are trying to estimate power

      local variables:
      current_power : static + dynamic power of core
      curr_f        : current frequency of core
      curr_v        : current voltage of core
      """
      current_power = self.energy_stats.power[('core', core)].d + self.energy_stats.power[('core', core)].s
      curr_f        = sim.dvfs.get_frequency(core)
      curr_v        = self.get_vdd_from_freq(curr_f)
      ratio         = (v*v*f) / (curr_v * curr_v * curr_f) * self.kappa
      return current_power * ratio


  def run_global_budget_allocator(self, time, time_delta, debug=False):
      """
      Algorithm 1 for Profit: Priority and Power/Performance Optimization for Many-Core Systems
      Function to allocate each core a power budget
      1. Calculate total budget used based on estimated power function
      2. Subtract from global budget allowed (95.0) and store as residual budget
      3. Build a max heap for each core based on the IPC
      4. Pop from heap, estimate a change in budget use based on estimated powers and update for that core
      5. repeat 4 while heap length, residual budget are greater than 0.
      """
      current_real_time = python_time.time()
      residual_budget   = 0
      consumed_budget   = 0
      core_ipc          = []
      for core in range(sim.config.ncores):
          self.core_budgets[core] = self.estimate_power(core=core, v=self.min_vdd, f=self.min_freq)
          consumed_budget         = consumed_budget + self.core_budgets[core]
          cycles = self.stats['time'][core].delta * sim.dvfs.get_frequency(core) / 1e9 # convert fs to cycles
          instrs = self.stats['coreinstrs'][core].delta
          ipc    = instrs / (cycles or 1)
          heapq.heappush(core_ipc, (-1 * ipc, core)) # have to use negative because heapq implements a min heap

      residual_budget = self.global_budget - consumed_budget
      if debug:
          print 'Global Available Budget, Residual Budget', self.global_budget, residual_budget
      while (residual_budget > 0) and (len(core_ipc) > 0):
          ipc, core = heapq.heappop(core_ipc)
          ipc       = -1 * ipc # revert the stored IPC back to a positive number (look at heappush call)
          if debug:
              print "Current Core, IPC", core, ipc
          delta = self.estimate_power(core=core, v=self.max_vdd, f=self.max_freq) - self.core_budgets[core]
          if delta <= residual_budget:
              self.core_budgets[core] = self.core_budgets[core] + delta
              residual_budget         = residual_budget - delta
          else:
              self.core_budgets[core] = self.core_budgets[core] + residual_budget
              residual_budget         = 0
      if debug:
          print "Budgets for each core:", self.core_budgets

  def get_state(self, time, time_delta, core, debug=False, scale=False):
      """
      Function to return the state for a specific core. The state
      is currently represented by: [ipc, mpki, current voltage, current power].
      """
      cycles           = self.stats['time'][core].delta * sim.dvfs.get_frequency(core) / 1e9 # convert fs to cycles
      num_instructions = self.stats['coreinstrs'][core].delta
      ipc              = num_instructions / (cycles or 1)
      mpki             = 0
      if num_instructions != 0:
          mpki = self.stats['misses'][core].delta / num_instructions * 1000

      current_voltage = self.get_vdd_from_freq(sim.dvfs.get_frequency(core))
      current_power   = self.energy_stats.power[('core', core)].d + self.energy_stats.power[('core', core)].s
      print "current_power state{}: ".format(core), current_power
      print "curr_ipc state{}: ".format(core), ipc
      #idle_cyles = self.stats['idle-time'][core].delta * sim.dvfs.get_frequency(core) / 1e9
      #dtlb_miss  = self.stats['dtlb-misses'][core].delta / 1748.
      #dtlb_access = self.stats['dtlb-accesses'][core].delta / 7509.
      #mem_stores = self.stats['memstores'][0].delta / 278.
      #mem_loads  = self.stats['memloads'][0].delta / 475.
      #l1d_stores  = self.stats['l1-dstores'][core].delta / 3604.
      #l1d_storemisses  = self.stats['l1-dstoremisses'][core].delta / 448.
      #l1d_loads  = self.stats['l1-dloads'][core].delta / 4505.
      #l1d_loadmisses  = self.stats['l1-dloadmisses'][core].delta / 458.

      #l2_stores = self.stats['l2stores'][core].delta / 448.
      #l2_storemisses = self.stats['l2storemiss'][core].delta / 448.
      #l2_loads = self.stats['l2loads'][core].delta / 459.
      #l2_loadmisses = self.stats['l2loadmiss'][core].delta / 459.

      #l3_stores = self.stats['l3stores'][core].delta / 448.
      #l3_storemisses = self.stats['l3storemisses'][core].delta / 391.
      #l3_loads = self.stats['l3loads'][core].delta / 459.
      #l3_loadmisses = self.stats['l3loadmisses'][core].delta / 73.




      idle_cyles = self.stats['idle-time'][core].delta * sim.dvfs.get_frequency(core) / 1e9
      dtlb_miss  = self.stats['dtlb-misses'][core].delta / 1.
      dtlb_access = self.stats['dtlb-accesses'][core].delta / 1.
      mem_stores = self.stats['memstores'][0].delta / 1.
      mem_loads  = self.stats['memloads'][0].delta / 1.
      l1d_stores  = self.stats['l1-dstores'][core].delta / 1.
      l1d_storemisses  = self.stats['l1-dstoremisses'][core].delta / 1.
      l1d_loads  = self.stats['l1-dloads'][core].delta / 1.
      l1d_loadmisses  = self.stats['l1-dloadmisses'][core].delta / 1.

      l2_stores = self.stats['l2stores'][core].delta / 1.
      l2_storemisses = self.stats['l2storemiss'][core].delta / 1.
      l2_loads = self.stats['l2loads'][core].delta / 1.
      l2_loadmisses = self.stats['l2loadmiss'][core].delta / 1.

      l3_stores = self.stats['l3stores'][core].delta / 1.
      l3_storemisses = self.stats['l3storemisses'][core].delta / 1.
      l3_loads = self.stats['l3loads'][core].delta / 1.
      l3_loadmisses = self.stats['l3loadmisses'][core].delta / 1.


      state = torch.tensor([ipc, mpki, current_voltage, current_power,
                            dtlb_miss, dtlb_access,
                            #mem_stores, mem_loads,
                            mem_loads,
                            l1d_stores, l1d_storemisses,
                            l1d_loads, l1d_loadmisses,
                            l2_stores, l2_storemisses,
                            l2_loads, l2_loadmisses,
                            l3_stores, l3_storemisses,
                            l3_loads, l3_loadmisses])

      state = (state - torch.mean(self.prev_data,0)) / (torch.std(self.prev_data,0) + self.eps)
      state = state.float()
      if debug:
          print "State.{} ".format(core), state
      return state


  def get_reward(self, time, time_delta, core, debug=False):
      """
      Function to return the reward for a specific core. The reward function is
      IPC -  PF * |current power - power budget|
      """
      cycles           = self.stats['time'][core].delta * sim.dvfs.get_frequency(core) / 1e9 # convert fs to cycles
      num_instructions = self.stats['coreinstrs'][core].delta
      ipc              = num_instructions / (cycles or 1)

      current_power   = self.energy_stats.power[('core', core)].d + self.energy_stats.power[('core', core)].s
      current_energy  = self.energy_stats.energy[('core', core, 'energy-static')]
      current_energy  += self.energy_stats.energy[('core', core, 'energy-dynamic')]
      print "current_power reward{}: ".format(core), current_power
      print "curr_ipc reward{}: ".format(core), ipc
      reward = ipc - self.PF * abs(current_power - self.core_budgets[core])
      #reward = ipc - current_energy
      if debug:
          print "Reward.{}".format(core), reward, current_power > self.core_budgets[core]
      return reward

  def run_local_RL_agent(self, time, time_delta, debug=False):
      """
      Function that perform forward propogation for our actor-critic model.
      1. iterate through each core
      2. get state and current reward
      3.
      """
      new_weights = []
      for core in range(sim.config.ncores):
          state  = self.get_state(time, time_delta, core, debug=True, scale=False)
          action = select_action(core, state)
          reward = self.get_reward(time, time_delta, core, debug=True)
          new_weights.append(action)
          models[core].rewards.append(reward)
      return new_weights

  def run_fast_dvfs(self, time, time_delta, new_weights, debug=False):
      """
      a FAST DVFS algorithm that can execute at the hardware level
      (and thus at nanosecond scale).
      1. Compute weighted power for each core
      2. compare weighted power with actual power
      For all cores:
      3. if weighted_power is less than actual power, increase frequency
      4. else if weighted power is greater than actual power, decrease frequency

      """
      if new_weights:
          # set weights to updated weights
          for i in range(len(new_weights)):
              self.curr_weights[i] = new_weights[i]

          # calculate total power consumption across all cores
      total_power = 0
      for core in range(sim.config.ncores):
          total_power = total_power + self.energy_stats.power[('core', core)].d + self.energy_stats.power[('core', core)].s

      # normalize the weights
      sum_weights = sum(self.curr_weights) + 0.0
      normalized_weights = []
      for i in range(len(self.curr_weights)):
          normalized_weights.append(self.curr_weights[i] / (sum_weights + 0.0))
      if debug:
          print "NORMALIZED WEIGHTS:", normalized_weights
      # update the frequencies of the cores based on weighted (allowed) powers
      weighted_powers = []
      for core in range(sim.config.ncores):
          current_power = self.energy_stats.power[('core', core)].d + self.energy_stats.power[('core', core)].s
          weighted_powers.append(normalized_weights[core] * total_power)
          current_freq  = sim.dvfs.get_frequency(core)

          if debug:
              print "Weighted Power.{}".format(core), weighted_powers[core]
              print "Actual Power.{}  ".format(core), current_power
              print "Curr Freq.{}     ".format(core), current_freq
          # if current power does not exceed the allowed power, increase the frequency
          if weighted_powers[core] > current_power:
              if current_freq < self.max_freq:
                  curr_idx = self.frequencies.index(current_freq)
                  sim.dvfs.set_frequency(core, self.frequencies[curr_idx - 1])

          # if current power exceeds the allowed power, reduce frequency
          elif weighted_powers[core] < current_power:
              if current_freq > self.min_freq:
                  curr_idx = self.frequencies.index(current_freq)
                  sim.dvfs.set_frequency(core, self.frequencies[curr_idx + 1])

  def get_stats(self, time, time_delta):
      self.save_stats[self.save_stats_idx] = {}
      for k,v in self.stats.items():
          num_values = len(v)
          self.save_stats[self.save_stats_idx][k] = [0]*num_values
          for i in range(num_values):
              self.save_stats[self.save_stats_idx][k][i] = v[i].delta
      self.save_stats_idx += 1

  def periodic(self, time, time_delta):

      ## COURSE GRAIN GLOBAL BUDGET ALLOCATOR
      if ((self.n_steps % self.M) == 0) and self.run_RL_agents:
          self.run_global_budget_allocator(time, time_delta, debug=True)
      ## Local RL Agent
      new_weights = []
      if ((self.n_steps % self.Q) == 0) and self.run_RL_agents:
          new_weights = self.run_local_RL_agent(time, time_delta, debug=True)


      if self.run_RL_agents:
          self.run_fast_dvfs(time, time_delta, new_weights, debug=True)

      if ((self.n_steps % self.K) == 0) and self.run_RL_agents:
          self.finish_episode()
      self.n_steps += 1
      #self.fd.write('\n')

  def finish_episode(self):
      print 'Total Elapsed Time', self.stats['time'][0].last
      print 'Total Elapsed Time', self.stats['time'][1].last


      all_losses = 0
      all_rewards = 0
      for core in range(sim.config.ncores):
          if len(models[core].saved_actions) > 1:

              R = 0
              saved_actions = models[core].saved_actions
              policy_losses = []
              value_losses  = []
              returns       = []

              iteration_number = 0
              for r in models[core].rewards[::-1]:
                  R = r + self.gamma * R
                  returns.insert(0, R)

              returns = torch.tensor(returns)
              print 'Discounted Rewards.{}'.format(core), returns[0]
              all_rewards += returns[0]
              returns = (returns - returns.mean()) / (returns.std() + self.eps)
              print 'NUMBER OF REWARDS:', len(models[core].rewards)
              for (log_prob, value), R in zip(saved_actions, returns):
                  advantage = R - value.item()

                  policy_losses.append(-1 * log_prob * advantage)
                  value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))


              optimizers[core].zero_grad()
              loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()
              print "LOSS.{}:".format(core), loss
              all_losses += loss
              loss.backward()
              optimizers[core].step()
              del models[core].rewards[:]
              del models[core].saved_actions[:]
              torch.save(models[core].state_dict(), WORKDIR + '/' + model_name.format(core))
      if all_losses:
          print "ALL LOSSES:  {}".format(all_losses)
          print "ALL REWARDS: {}".format(all_rewards)

sim.util.register(RLAgent())
