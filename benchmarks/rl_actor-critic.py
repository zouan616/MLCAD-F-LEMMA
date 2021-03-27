import os
import sys
import sim
import heapq
import random
import operator
import itertools

from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


x = np.array([0,1,1,2])
print("X:", x)
NUM_ACTIONS = len([(2000, 1.2), (1800, 1.08), (1600, 0.96), (1400, 0.84)])
# some sources:
# https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
# https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        n_nodes = 32
        self.affine1 = nn.Linear(4, n_nodes)

        # actor's layer
        self.action_head = nn.Linear(n_nodes, NUM_ACTIONS)

        # critic's layer
        self.value_head = nn.Linear(n_nodes, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)



class Power:
  def __init__(self, static, dynamic):
    # print("We are here.")
    self.s = static
    self.d = dynamic
  def __add__(self, v):
    return Power(self.s + v.s, self.d + v.d)
  def __sub__(self, v):
    return Power(self.s - v.s, self.d - v.d)

def build_dvfs_table(tech):
  # Build a table of (frequency, voltage) pairs.
  # Frequencies should be from high to low, and end with zero (or the lowest possible frequency)
  if tech == 22:
    return [ (2000, 1.0), (1800, 0.9), (1500, 0.8), (1000, 0.7), (0, 0.6) ]
  elif tech == 45:
    return [(2000, 1.2), (1800, 1.08), (1600, 0.96), (1400, 0.84)]
    # from the profit paper
    return [(2800, 0.95), (2600, 0.85), (2400, 0.75), (2200, 0.65)]
    # return [(2000, 1.2), (1800, 1.1), (1600, 1.0), (1500, 0.9), (1300, 0.8), (1000, 0.6)]
    # # return [ (2000, 1.0), (1000, 1.0)]

  else:
    raise ValueError('No DVFS table available for %d nm technology node' % tech)

# O(n) but doesnt matter since lists are small.
def find_closest_value(list, number):
  return min(list, key = lambda x:abs(x-number))

class EnergyStats:
  def setup(self, args):
    args = dict(enumerate((args or '').split(':')))
    # interval_ns = long(args.get(0, None) or 1000000) # Default power update every 1 ms
    interval_ns = long(args.get(0,None) or 500000)
    interval_ns = long(args.get(0,None) or 1000)
    sim.util.Every(interval_ns * sim.util.Time.NS, self.periodic, roi_only = True)
    self.dvfs_table = build_dvfs_table(int(sim.config.get('power/technology_node')))
    #
    self.name_last = None
    self.time_last_power = 0
    self.time_last_energy = 0
    self.in_stats_write = False
    self.power = {}
    self.energy = {}
    for metric in ('energy-static', 'energy-dynamic'):
      for core in range(sim.config.ncores):
        sim.stats.register('core', core, metric, self.get_stat)
        sim.stats.register('L1-I', core, metric, self.get_stat)
        sim.stats.register('L1-D', core, metric, self.get_stat)
        sim.stats.register('L2', core, metric, self.get_stat)
      #sim.stats.register_per_thread('core-'+metric, 'core', metric)
      #sim.stats.register_per_thread('L1-I-'+metric, 'L1-I', metric)
      #sim.stats.register_per_thread('L1-D-'+metric, 'L1-D', metric)
      #sim.stats.register_per_thread('L2-'+metric, 'L2', metric)
      sim.stats.register('processor', 0, metric, self.get_stat)
      sim.stats.register('dram', 0, metric, self.get_stat)

  def periodic(self, time, time_delta):
    self.update()

  def hook_pre_stat_write(self, prefix):
    if not self.in_stats_write:
      self.update()

  def hook_sim_end(self):
    if self.name_last:
      sim.util.db_delete(self.name_last, True)

  def update(self):
    if sim.stats.time() == self.time_last_power:
      # Time did not advance: don't recompute
      return
    if not self.power or (sim.stats.time() - self.time_last_power >= 10 * sim.util.Time.US):
      # Time advanced significantly, or no power result yet: compute power
      #   Save snapshot
      current = 'energystats-temp%s' % ('B' if self.name_last and self.name_last[-1] == 'A' else 'A')
      self.in_stats_write = True
      sim.stats.write(current)
      self.in_stats_write = False
      #   If we also have a previous snapshot: update power
      if self.name_last:
        power = self.run_power(self.name_last, current)
        self.update_power(power)
      #   Clean up previous last
      if self.name_last:
        sim.util.db_delete(self.name_last)
      #   Update new last
      self.name_last = current
      self.time_last_power = sim.stats.time()
    # Increment energy
    self.update_energy()

  def get_stat(self, objectName, index, metricName):
    if not self.in_stats_write:
      self.update()
    return self.energy.get((objectName, index, metricName), 0L)

  def update_power(self, power):
    # print "UPDATE POWER"
    def get_power(component, prefix = ''):
      # print("Hello.")
      # print("SUBTHRESHOLD ", component[prefix + 'Subthreshold Leakage'])
      # print("GATETHRESHOLD", component[prefix + 'Gate Leakage'])
      return Power(component[prefix + 'Subthreshold Leakage'] + component[prefix + 'Gate Leakage'], component[prefix + 'Runtime Dynamic'])
    for core in range(sim.config.ncores):
      self.power[('L1-I', core)] = get_power(power['Core'][core], 'Instruction Fetch Unit/Instruction Cache/')
      self.power[('L1-D', core)] = get_power(power['Core'][core], 'Load Store Unit/Data Cache/')
      self.power[('L2',   core)] = get_power(power['Core'][core], 'L2/')
      self.power[('core', core)] = get_power(power['Core'][core]) - (self.power[('L1-I', core)] + self.power[('L1-D', core)] + self.power[('L2', core)])
    self.power[('processor', 0)] = get_power(power['Processor'])
    self.power[('dram', 0)] = get_power(power['DRAM'])

  def update_energy(self):
    # print "UPDATE ENERGY"
    if self.power and sim.stats.time() > self.time_last_energy:
      time_delta = sim.stats.time() - self.time_last_energy
      for (component, core), power in self.power.items():
        self.energy[(component, core, 'energy-static')] = self.energy.get((component, core, 'energy-static'), 0) + long(time_delta * power.s)
        self.energy[(component, core, 'energy-dynamic')] = self.energy.get((component, core, 'energy-dynamic'), 0) + long(time_delta * power.d)
        if component == 'core':
            print str(core) + ".STATIC POWER ", power.s
            print str(core) + ".DYNAMIC POWER", power.d
            print str(core) + ".STATIC ENERGY", self.energy[(component, core, 'energy-static')]
            print str(core) + ".DYNAMIC ENERGY", self.energy[(component, core, 'energy-dynamic')]
      self.time_last_energy = sim.stats.time()

  def get_vdd_from_freq(self, f):
    # Assume self.dvfs_table is sorted from highest frequency to lowest
    for _f, _v in self.dvfs_table:
      if f >= _f:
        return _v
    assert ValueError('Could not find a Vdd for invalid frequency %f' % f)

  def gen_config(self, outputbase):
    # print "WRITING CONFIGURATION FILE"
    freq = [ sim.dvfs.get_frequency(core) for core in range(sim.config.ncores) ]
    vdd = [ self.get_vdd_from_freq(f) for f in freq ]
    # print(vdd)
    configfile = outputbase+'.cfg'
    cfg = open(configfile, 'w')
    cfg.write('''
[perf_model/core]
frequency[] = %s
[power]
vdd[] = %s
    ''' % (','.join(map(lambda f: '%f' % (f / 1000.), freq)), ','.join(map(str, vdd))))
    cfg.close()
    return configfile

  def run_power(self, name0, name1):
    print "RUNNING POWER"
    outputbase = os.path.join(sim.config.output_dir, 'energystats-temp')

    configfile = self.gen_config(outputbase)

    os.system('unset PYTHONHOME; %s -d %s -o %s -c %s --partial=%s:%s --no-graph --no-text' % (
      os.path.join(os.getenv('SNIPER_ROOT'), 'tools/mcpat.py'),
      sim.config.output_dir,
      outputbase,
      configfile,
      name0, name1
    ))

    result = {}
    execfile(outputbase + '.py', {}, result)
    return result['power']





# above is energystats.py






class RLAgent:

  def setup(self, args):
    self.run_RL_agents = True
    self.reward_function = 0
    self.peak_IPC        = 2.0
    self.peak_Energy     = 5.0 #Watts

    # setup energystats object: we will get the
    # current dynamic power from this object
    self.energy_stats = EnergyStats()
    sim.util.register(self.energy_stats)
    self.avg_ipcs   = []   # running average of ipc over all cores
    self.avg_mpki   = []   # running average of mpki over all cores
    self.start_Vdd  = 0.85 # starting Vdd for each core
    self.start_freq = 2600 # starting frequency for each core
    # updated values to work with McPat
    self.start_Vdd  = 0.75
    self.start_freq = 1500
    self.n_steps    = 0    # n_steps taken
    self.q_table    = {}   # will hold our Q-values
    self.theta      = 0.1  # learning rate for Q-learning
    self.gamma      = 0.9  # future discounted reward rate
    self.epsilon    = 0.1  # exploration probability for epsilon-greedy

    # all four components of the state space
    self.IPCs    = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    self.MPKIs   = [0, 5, 10, 15, 20, 25]
    self.VFs     = [0.95, 0.85, 0.75, 0.65]
    self.VFs     = [1.2, 1.1, 1.0, 0.9]
    self.PVs     = [0, 2, 4, 6, 8]
    self.actions = [2800, 2600, 2400, 2200] # actions we can take
    self.actions = [2000, 1800, 1500, 1000]


    self.M             = 15    # global budget allocator will execute every M epochs.
    self.global_budget = 95.0  # total power budget available
    self.core_budgets  = []
    self.PF            = 5.0   # penalty factor
    self.kappa         = 0.9   # transition cost of VF level


    # produces cartesian product given components of state space
    state_space = list(itertools.product(self.IPCs, self.MPKIs, self.VFs, self.PVs))

    # set up q-table and init frequency of each core.
    for core in range(sim.config.ncores):
      self.avg_ipcs.append(0)
      self.avg_mpki.append(0)
      self.core_budgets.append(0)
      sim.dvfs.set_frequency(core, self.start_freq)
      self.q_table[core] = {}
      for state in state_space:
        self.q_table[core][state] = {}
        for action in self.actions:
          self.q_table[core][state][action] = 0


    self.dvfs_table = build_dvfs_table(int(sim.config.get('power/technology_node')))

    # this is for later. once we learn how to discretize the state space.
    # self.IPCs  = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    # self.MPKIs = [0, 5, 10, 15, 20, 25, 30]
    # self.VFS   = [2200, 2400, 2600, 2800]
    # self.PVs   = [0, 2, 4, 6, 8]
    args = dict(enumerate((args or '').split(':')))
    filename = args.get(0, None)
    interval_ns = long(args.get(1,500000))
    interval_ns = long(args.get(1,500000))
    interval_ns = long(args.get(0,None) or 1000)

    print "Interval:", interval_ns, "ns"
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
      current_power = self.energy_stats.power[('core', core)].d
      curr_f        = sim.dvfs.get_frequency(core)
      curr_v        = self.get_vdd_from_freq(curr_f)
      ratio         = (v*v*f) / (curr_v * curr_v * curr_f) * self.kappa
      return current_power * ratio


  def periodic(self, time, time_delta):
    self.n_steps += 1

    ## COURSE GRAIN GLOBAL BUDGET ALLOCATOR
    if self.n_steps % self.M == 0 or self.n_steps == 2 and self.run_RL_agents:
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


    if self.isTerminal:
      self.fd.write('[IPC MPKI] ')
    self.fd.write('%u ns\n' % (time / 1e6)) # Time in ns
    qq = 0

    ## FINE GRAIN DISTRIBUTED Q-LEARNING AGENT
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
      self.avg_ipcs[qq] = (self.n_steps * self.avg_ipcs[qq] + ipc) / (self.n_steps + 1)
      self.avg_mpki[qq] = (self.n_steps * self.avg_mpki[qq] + mpki) / (self.n_steps + 1)

      if self.run_RL_agents:
      # discretize current properties
        curr_discrete_ipc  = find_closest_value(self.IPCs, ipc)
        curr_discrete_mpki = find_closest_value(self.MPKIs, mpki)
        curr_discrete_v    = self.get_vdd_from_freq(sim.dvfs.get_frequency(core))
        curr_discrete_p    = find_closest_value(self.PVs, self.energy_stats.power[('core', core)].d)

        # currently setting reward to some measure of throughput.

        if self.reward_function == 0:
            reward = ipc
            reward = reward - self.PF * abs(curr_discrete_p - self.core_budgets[core])
        
        # Option 1: throughput - energy - power overshoot
        if self.reward_function == 1:
            reward = ipc / self.peak_IPC  - (self.energy_stats.power[('core', core)].d * self.stats['time'][core].delta) / self.peak_Energy
            reward = reward - self.PF * abs(curr_discrete_p - self.core_budgets[core])

        # Option 2: throughput / energy - power overshoot
        if self.reward_function == 2:
            throughput_value = ipc / self.peak_IPC 
            energy_value     = (self.energy_stats.power[('core', core)].d * self.stats['time'][core].delta) / self.peak_Energy
            reward = throughput_value / energy_value - self.PF * abs(curr_discrete_p - self.core_budgets[core])

        # reward = ipc
        # reward = reward - self.PF * abs(curr_discrete_p - self.core_budgets[core])
        print "Curr State -", core, ":", curr_discrete_ipc, curr_discrete_mpki, curr_discrete_v, curr_discrete_p
        curr_state = (curr_discrete_ipc, curr_discrete_mpki, curr_discrete_v, curr_discrete_p)

        # perform update of Q-values

        max_q_next = max(self.q_table[core][curr_state].iteritems(), key=operator.itemgetter(1))[1]
        # print "Updating Q Table!"
        self.q_table[core][curr_state][sim.dvfs.get_frequency(core)] += self.theta * (reward + self.gamma * max_q_next - self.q_table[core][curr_state][sim.dvfs.get_frequency(core)])

        # epsilon-greedy policy
        if random.random() <= self.epsilon:
          # print "Random Action!"
          random_action = random.choice(self.actions)
          # print random_action
          curr_action = random_action
        else:
          # print "Max Action!"
          max_action    = max(self.q_table[core][curr_state].iteritems(), key=operator.itemgetter(1))[0]
          # print max_action
          curr_action = max_action

        sim.dvfs.set_frequency(core,curr_action)

    self.fd.write('\n')


sim.util.register(RLAgent())
# sim.util.register(EnergyStats())