#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

from common import create_3r_and_init, get_usual_observers

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
from LQPctrl.task import JointTask
from LQPctrl.task_ctrl import KpCtrl
tasks = []
goal = .1
tasks.append(JointTask(joints["Shoulder"], KpCtrl(goal, 10), [], 1., 0, True))


## EVENTS
events = []


from LQPctrl.LQPctrl import LQPcontroller
gforcemax = {"Elbow":5,"Wrist":2} #"Shoulder":10,
# ...or...
#gforcemax = {"Shoulder":0,"Elbow":5,"Wrist":2}

lqpc = LQPcontroller(gforcemax, tasks=tasks)
w.register(lqpc)


############################
#                          #
# SET OBSERVERS & SIMULATE #
#                          #
############################
obs = get_usual_observers(w)

from common import RecordJointPosition
obs.append(RecordJointPosition(joints["Shoulder"]))

## SIMULATE
from numpy import arange
from arboris.core import simulate
simulate(w, arange(0,5,0.01), obs)


## RESULTS
print("end of the simulation")

import pylab as pl
pl.plot(obs[-1].get_record())
xlim = pl.xlim()
pl.plot(xlim, [.1,.1], 'r:')
pl.ylim([0, .6])
pl.ylabel("position (rad)")
pl.xlabel("step")
pl.title("Shoulder Joint Evolution")
pl.show()

