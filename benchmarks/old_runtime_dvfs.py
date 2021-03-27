"""
ipctrace.py

Write a trace of instantaneous IPC values for all cores.
First argument is either a filename, or none to write to standard output.
Second argument is the interval size in nanoseconds (default is 10000)
"""

import sys, os, sim
# import numpy as np
counter = 0
class IpcTrace:
  def setup(self, args):
    args = dict(enumerate((args or '').split(':')))
    filename = args.get(0, None)
    # interval_ns = long(args.get(1, 10000))
    interval_ns = long(args.get(1,500))
    print "Interval:", interval_ns
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
    }
    sim.util.Every(interval_ns * sim.util.Time.MS, self.periodic, statsdelta = self.sd, roi_only = True)

  def periodic(self, time, time_delta):
    global counter
    if counter == 5:
        for core in range(sim.config.ncores):
            sim.dvfs.set_frequency(core, 3000) ## Number is in MHz
    if counter == 0:
        for core in range(sim.config.ncores):
            sim.dvfs.set_frequency(core, 1000)
    counter += 1
    for core in range(sim.config.ncores):
        print("DVFS per Core:", sim.dvfs.get_frequency(core))
    print "DVFS:", sim.dvfs
    if self.isTerminal:
      self.fd.write('[IPC] ')
    self.fd.write('%u ns' % (time / 1e6)) # Time in ns
    for core in range(sim.config.ncores):
      # detailed-only IPC
      cycles = (self.stats['time'][core].delta - self.stats['ffwd_time'][core].delta) * sim.dvfs.get_frequency(core) / 1e9 # convert fs to cycles
      instrs = self.stats['instrs'][core].delta
      ipc = instrs / (cycles or 1) # Avoid division by zero
      #self.fd.write(' %.3f' % ipc)

      # include fast-forward IPCs
      cycles = self.stats['time'][core].delta * sim.dvfs.get_frequency(core) / 1e9 # convert fs to cycles
      instrs = self.stats['coreinstrs'][core].delta
      ipc = instrs / (cycles or 1)
      self.fd.write(' %.3f' % ipc)
    self.fd.write('\n')


sim.util.register(IpcTrace())
