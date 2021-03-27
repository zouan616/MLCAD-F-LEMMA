import os
import sim

class Power:
  def __init__(self, static, dynamic):
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

  else:
    raise ValueError('No DVFS table available for %d nm technology node' % tech)

class EnergyStats:
  def setup(self, args):
    args = dict(enumerate((args or '').split(':')))
    # interval_ns = long(args.get(0, None) or 1000000) # Default power update every 1 ms
    interval_ns = long(args.get(0,None) or 500000)
    interval_ns = long(args.get(0,None) or 1000)
    interval_ns = long(1000)
    sim.util.Every(interval_ns * sim.util.Time.NS, self.periodic, roi_only = True)
    self.dvfs_table = build_dvfs_table(int(sim.config.get('power/technology_node')))
    print "EnergyStats Interval:", interval_ns
    self.name_last = None
    self.time_last_power = 0
    self.time_last_energy = 0
    self.in_stats_write = False
    self.power = {}
    self.energy = {}
    sim.hooks.register(sim.hooks.HOOK_SIM_END, self.print_stats)

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

  def print_stats(self):
      time_delta = sim.stats.time() - self.time_last_energy
      for (component, core), power in self.power.items():
          self.energy[(component, core, 'energy-static')] = self.energy.get((component, core, 'energy-static'), 0) + long(1000 * power.s)
          self.energy[(component, core, 'energy-dynamic')] = self.energy.get((component, core, 'energy-dynamic'), 0) + long(1000 * power.d)
          if component == 'core':
              print str(core) + ".STATIC POWER (W) AFTER", power.s
              print str(core) + ".DYNAMIC POWER (W) AFTER", power.d
              print str(core) + ".STATIC ENERGY (nJ) AFTER", self.energy[(component, core, 'energy-static')]
              print str(core) + ".DYNAMIC ENERGY (nJ) AFTER", self.energy[(component, core, 'energy-dynamic')]
              print "time_delta", time_delta

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
          self.energy[(component, core, 'energy-static')] = self.energy.get((component, core, 'energy-static'), 0) + long(1000 * power.s)
          self.energy[(component, core, 'energy-dynamic')] = self.energy.get((component, core, 'energy-dynamic'), 0) + long(1000 * power.d)
          if component == 'core':
              print str(core) + ".STATIC POWER (W)", power.s
              print str(core) + ".DYNAMIC POWER (W)", power.d
              print str(core) + ".STATIC ENERGY (nJ)", self.energy[(component, core, 'energy-static')]
              print str(core) + ".DYNAMIC ENERGY (nJ)", self.energy[(component, core, 'energy-dynamic')]
          #    print "time_delta", time_delta

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
    # print "RUNNING POWER"
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
