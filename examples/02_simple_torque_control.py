#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

from common import create_3r_and_init, get_usual_observers, print_lqp_perf

#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_3r_and_init(gpos=(.5,.5,.5))
joints = w.getjoints()


#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import TorqueTask
from LQPctrl.task_ctrl import ValueCtrl
tasks = []
goal = .1
tasks.append(TorqueTask(joints["Shoulder"], ValueCtrl(goal), [], 1., 0, True))


## EVENTS
events = []


## LQP CONTROLLER
from LQPctrl.LQPctrl import LQPcontroller
gforcemax = {"Shoulder":10,"Elbow":5,"Wrist":2}

lqpc = LQPcontroller(gforcemax, tasks=tasks)
w.register(lqpc)


############################
#                          #
# SET OBSERVERS & SIMULATE #
#                          #
############################
obs = get_usual_observers(w)

from common import RecordGforce
obs.append(RecordGforce(lqpc))

## SIMULATE
from numpy import arange
from arboris.core import simulate
simulate(w, arange(0,3,0.01), obs)


###########
#         #
# RESULTS #
#         #
###########
print("end of the simulation")
print_lqp_perf(lqpc)

import pylab as pl
pl.plot(obs[-1].get_record())
pl.ylim([0,.15])
pl.ylabel("Gforce (N.m)")
pl.xlabel("step")
pl.title("Shoulder Joint Evolution")
pl.show()

