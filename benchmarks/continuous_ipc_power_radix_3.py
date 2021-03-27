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
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical


import time as python_time
np.random.seed(42)
torch.manual_seed(42)


WORKDIR               = os.environ['BENCHMARKS_ROOT'] #current directory
CURR_BENCHMARK        = os.environ['CURR_BENCHMARK']
REL_SAMPLED_DATA_PATH = 'sample-data-' + CURR_BENCHMARK + '.pt'
ABS_SAMPLED_DATA_PATH = WORKDIR + '/' + REL_SAMPLED_DATA_PATH
sys.path.append(WORKDIR)




import final_pacman_power as pacman_power_checker
import continuous_network_ipc_power as single_pacman_network

def select_action(core, state):
    mus,sigmas, state_value = models[core](state)

    sigmas_ = torch.zeros(2,2)
    sigmas_[0,0] = sigmas[0]
    sigmas_[1,1] = sigmas[1]
    m = MultivariateNormal(mus, sigmas_)
    action = m.sample()
    #action2 = m.sample()
    models[core].saved_actions.append(single_pacman_network.SavedAction(m.log_prob(action), state_value))

    # clip so that the output of the sample is at least a small positive value.
    return action.numpy()
    #return np.maximum(0,np.minimum(1,action.numpy()))


models     = []
optimizers = []
model_name = 'network_energy_correlation_radix_3333_{}_{}.pt'
for core in range(sim.config.ncores):
    print 'Creating an actor-critic model for core number: ' + str(core)
    models.append(single_pacman_network.Policy())
    if model_name.format(CURR_BENCHMARK, core) in os.listdir(WORKDIR):
        print 'Loading previously trained model'
        models[core].load_state_dict(torch.load(WORKDIR + '/' + model_name.format(CURR_BENCHMARK, core)))
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
      self.outs = []
      self.avg_powers     = [0] * sim.config.ncores
      self.avg_ipcs       = [0] * sim.config.ncores

      self.prev_energies = [0] * sim.config.ncores
      self.max_power     = 15.0
      self.max_ipc       = 5.0


      self.prev_data    = torch.load(ABS_SAMPLED_DATA_PATH)
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
      self.n_steps       = 0 # simulation steps
      self.fast_steps    = 0
      self.eps           = np.finfo(np.float32).eps.item() # for numerical stability
      self.global_budget = 95.0 # total power budget available
      self.PF            = 0.0    # penalty factor in reward function




      self.kappa         = 0.9  # transition cost from moving VF levels
      self.energy_weight = 1.0 # energy weight
      self.PF2           = 5.0 # ipc weight
      # unsure if we need these
      self.run_RL_agents = True # run learning based method
      self.gamma         = 0.95
      self.max_delta_energy = 6539505.

      # setup initial weights, budget, and set frequency of each core.
      # 3 -> lowest VF  (1400 MHz, 0.84 Volts)
      # 0 -> highest VF (2000 MHz, 1.20 Volts)
      self.start_idx   = 0
      self.start_freq  = self.frequencies[self.start_idx]
      self.start_vdd          = self.voltages[self.start_idx]

      self.vector_frequencies = [[]] * sim.config.ncores
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

      self.high_threshold = 8.0
      self.mid_threshold  = 5.0
      self.low_threshold  = 2.0


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
      state[0] = 0.0 #IPC. 
      state[1] = 0.0 #mpki. 
      state[2] = 0.0
      state[3] = 0.0
      state[4] = 0.0
      state[5] = 0.0
      state[6] = 0.0
      #state[7] = 0.0
      #state[8] = 0.0
      #state[9] = 0.0
      #state[10] = 0.0
      #state[11] = 0.0
      #state[12] = 0.0
      #state[13] = 0.0
      #state[14] = 0.0
      #state[15] = 0.0
      #state[16] = 0.0
      #state[17] = 0.0
      #state[18] = 0.0
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

      curr_delta_energy        = current_energy - self.prev_energies[core]
      self.prev_energies[core] = current_energy
      reward = -1 * curr_delta_energy / self.max_delta_energy * self.energy_weight
      print "energy reward term.{} {}".format(core, reward)
      if self.vector_frequencies[core]:
          print "ipc reward term.{} {}".format(core, np.mean(self.vector_frequencies[core])/self.max_freq * self.PF2 * ipc)
          reward = reward + np.mean(self.vector_frequencies[core])/self.max_freq * self.PF2 * ipc
      return reward

  def run_local_RL_agent(self, time, time_delta, debug=False):
      """
      Function that perform forward propogation for our actor-critic model.
      1. iterate through each core
      2. get state and current reward
      3.
      """
      ideal_vfs = []
      for core in range(sim.config.ncores):
          state  = self.get_state(time, time_delta, core, debug=True, scale=False)
          action = select_action(core, state)
          reward = self.get_reward(time, time_delta, core, debug=True)
          ideal_vfs.append(action)
          models[core].rewards.append(reward)
      return ideal_vfs

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
      # THIS IS THE CORRECT ONE.
      self.fast_steps += 1
      for core in range(sim.config.ncores):
          cycles = self.stats['time'][core].delta * sim.dvfs.get_frequency(core) / 1e9 # convert fs to cycles
          instrs = self.stats['coreinstrs'][core].delta
          ipc = instrs / (cycles or 1) # current IPC

          self.avg_ipcs[core] = (self.fast_steps * self.avg_ipcs[core] + ipc) / (self.fast_steps + 1)
          current_power   = self.energy_stats.power[('core', core)].d + self.energy_stats.power[('core', core)].s
          self.avg_powers[core] = (self.fast_steps * self.avg_powers[core] + current_power) / (self.fast_steps + 1)
          print "ipc, threshold, avg.{}".format(core), ipc, new_weights[core][1], self.avg_ipcs[core]
          print "power, threshold, avg.{}".format(core), current_power, new_weights[core][0], self.avg_powers[core]
          dot_product = ((current_power / self.max_power) * new_weights[core][0] + (ipc / self.max_ipc) * new_weights[core][1])
          #self.frequencies  = [2000, 1800, 1600, 1400]
          avg_sum = (self.avg_ipcs[core] / self.max_ipc) + (self.avg_powers[core] / self.max_power)

          if dot_product > avg_sum * 1.0:
              curr_freq = sim.dvfs.get_frequency(core)
              if curr_freq != 2000:
                  sim.dvfs.set_frequency(core, self.frequencies[0]) #highest
          elif dot_product > avg_sum * 0.6:
              curr_freq = sim.dvfs.get_frequency(core)
              if curr_freq != 1800:
                  sim.dvfs.set_frequency(core, self.frequencies[1])
          elif dot_product > avg_sum * 0.2:
              curr_freq = sim.dvfs.get_frequency(core)
              if curr_freq != 1600:
                  sim.dvfs.set_frequency(core, self.frequencies[2])
          else:
              curr_freq = sim.dvfs.get_frequency(core)
              if curr_freq != 1400:
                  sim.dvfs.set_frequency(core, self.frequencies[3])

          self.vector_frequencies[core].append(sim.dvfs.get_frequency(core))





        #  if (current_power / self.max_power * new_weights[core][0] + ipc / self.max_ipc * new_weights[core][1]) > 1/2.(self.avg_powers[core] / self.max_power + self.avg_ipcs[core] / self.max_ipc):
        #      curr_freq = sim.dvfs.get_frequency(core)
        #      idx = self.frequencies.index(curr_freq)
        #      if curr_freq < self.max_freq:
        #          sim.dvfs.set_frequency(core, self.frequencies[idx - 1])
         # else:
        #      curr_freq = sim.dvfs.get_frequency(core)
        #      idx = self.frequencies.index(curr_freq)
        #      if curr_freq > self.min_freq:
        #          sim.dvfs.set_frequency(core, self.frequencies[idx + 1])



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
      if ((self.n_steps % self.Q) == 0) and self.run_RL_agents:
          self.outs = self.run_local_RL_agent(time, time_delta, debug=True)
          self.vector_frequencies = [[]] * sim.config.ncores
          for core in range(sim.config.ncores):
              print "weight_ipc.{}".format(core), self.outs[core][1]
              print "weight_power.{}".format(core), self.outs[core][0]

          #print("output weights:\n")
          #print(self.outs)

      self.run_fast_dvfs(time, time_delta, self.outs, debug=True)
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
              torch.save(models[core].state_dict(), WORKDIR + '/' + model_name.format(CURR_BENCHMARK, core))
      if all_losses:
          print "ALL LOSSES:  {}".format(all_losses)
          print "ALL REWARDS: {}".format(all_rewards)

sim.util.register(RLAgent())
