#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=25 may 2011

from common import create_3r_and_init, get_usual_observers, print_lqp_perf

#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_3r_and_init(gpos=(.5,.4,.3))
joints = w.getjoints()


#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import MultiJointTask
from LQPctrl.task_ctrl import KpCtrl
tasks = []
goal = [.1,.1,.1]
tasks.append(MultiJointTask(joints, KpCtrl(goal, 10), [], 1., 0, True))


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

from common import RecordJointPosition
obs.append(RecordJointPosition(joints))

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
xlim = pl.xlim()
pl.plot(xlim, [.1,.1], 'r:')
pl.ylim([0, .6])
pl.ylabel("position (rad)")
pl.xlabel("step")
pl.title("Arm Joints Evolution")
pl.legend(["Shoulder","Elbow","Wrist"])
pl.show()

