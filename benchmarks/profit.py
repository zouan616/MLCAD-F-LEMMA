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




import final_pacman_power as pacman_power_checker
actions = [2000, 1800, 1600, 1400]




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
      self.max_delta_energy = 6539505.

      self.prev_energies = [0] * sim.config.ncores


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
      self.n_steps       = -1 # simulation steps
      self.eps           = np.finfo(np.float32).eps.item() # for numerical stability
      self.global_budget = 95.0 # total power budget available
      self.PF            = 5.0    # penalty factor in reward function
      self.kappa         = 0.9  # transition cost from moving VF levels
      self.epsilon       = 0.1


      self.discrete_IPCs   = np.array([0, 1.0, 1.8])
      self.discrete_MPKIs  = np.array([0, 5.8, 14.])
      self.discrete_VFs    = np.copy(self.voltages)
      self.discrete_POWERs = np.array([2.0, 5.3, 9.])

      self.num_ipcs     = len(self.discrete_IPCs)
      self.num_mpkis    = len(self.discrete_MPKIs)
      self.num_powers   = len(self.discrete_POWERs)
      self.num_voltages = len(self.voltages)
      self.num_actions  = len(self.frequencies)

      self.q_table = np.zeros((sim.config.ncores,self.num_ipcs, self.num_mpkis, self.num_powers, self.num_voltages, self.num_actions))
      if "q-table-{}.npy".format(CURR_BENCHMARK) in os.listdir(WORKDIR):
          self.q_table = np.load(WORKDIR + '/' + "q-table-{}.npy".format(CURR_BENCHMARK))
          print "Loading previously saved Q-table"
      else:
          print "Creating Q-table with zeros."

      # unsure if we need these
      self.run_RL_agents = True # run learning based method
      self.gamma         = 0.95
      self.learning_rate = 0.1

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

      self.curr_states = []
      for core in range(sim.config.ncores):


          ipc_idx = self.get_closest_idx(0, self.discrete_IPCs)
          mpki_idx = self.get_closest_idx(0, self.discrete_MPKIs)
          power_idx = self.get_closest_idx(0, self.discrete_POWERs)
          volt_idx = self.get_closest_idx(0, self.discrete_VFs)
          self.curr_states.append(np.array([ipc_idx, mpki_idx, power_idx, volt_idx]))



  def select_action(self, core, state):

      if random.random() < 0.1:
          action = np.random.choice(actions)
      else:
          action_idx = np.argmax(self.q_table[core, state[0], state[1], state[2], state[3]])
          action = actions[action_idx]

      return action

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
      4. Pop from heap, estimate a core)].d + self.energy_stats.power[('core', core)].s

      ipc_idx = self.get_closest_idx(ipc, self.discrete_IPCs)
      mpki_idx = self.get_closest_idx(mpki, self.discrete_MPKIs)
      volt_idx = self.get_closest_idx(current_voltage, self.discrete_VOLTAGEschange in budget use based on estimated powers and update for that core
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

  def get_closest_idx(self, value, array):
      """
      function to discretize value by return index to item closest
      to 'value' in 'array'
      """
      return np.argmin(np.abs(value - array))

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

      ipc_idx = self.get_closest_idx(ipc, self.discrete_IPCs)
      mpki_idx = self.get_closest_idx(mpki, self.discrete_MPKIs)
      power_idx = self.get_closest_idx(current_power, self.discrete_POWERs)
      volt_idx = self.get_closest_idx(current_voltage, self.discrete_VFs)
      return np.array([ipc_idx, mpki_idx, power_idx, volt_idx])


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
      print "CURRENT ENERGY IN REWARD FUNCTION.{}:".format(core), current_energy
      reward = ipc - self.PF * abs(current_power - self.core_budgets[core])
      return reward



  def run_local_RL_agent(self, time, time_delta, debug=False):
      """
      Function that perform forward propogation for our actor-critic model.
      1. iterate through each core
      2. get state and current reward
      3.
      """
      for core in range(sim.config.ncores):
          prev_state            = self.curr_states[core]
          state                 = self.get_state(time, time_delta, core, debug=True, scale=False)
          self.curr_states[core] = state
          reward    = self.get_reward(time, time_delta, core, debug=True)
          print("idx", state)
          max_next_q = np.max(self.q_table[core, state[0], state[1], state[2], state[3],:])
          curr_q_value = self.q_table[core,prev_state[0], prev_state[1], prev_state[2], prev_state[3]]
          self.q_table[core,prev_state[0], prev_state[1], prev_state[2], prev_state[3]] = \
                   curr_q_value + self.learning_rate * (reward + self.gamma *  max_next_q - curr_q_value)

          action                = self.select_action(core, state)
          sim.dvfs.set_frequency(core, action)



          sim.dvfs.set_frequency(core, action)

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
          new_weights = self.run_local_RL_agent(time, time_delta, debug=True)

    #  if ((self.n_steps % self.K) == 0) and self.run_RL_agents:
    #      self.finish_episode()
      self.n_steps += 1
      #self.fd.write('\n')

  def finish_episode(self):
      np.save(WORKDIR + '/' + "q-table-{}.npy".format(CURR_BENCHMARK), self.q_table)




sim.util.register(RLAgent())
