import math
import random
# import sniper_lib
import sys
import os
print os.getcwd()
import energystats
# import pccalc
import sim_hooks
import sim
print sim.config.ncores

q = 1
def foo(t):
    global q
    print "Number of ticks", q
    q += 1

sim_hooks.register(sim_hooks.HOOK_PERIODIC,foo)
