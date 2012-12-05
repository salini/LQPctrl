#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=25 may 2011

from common import create_3r_and_init, get_usual_observers

#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_3r_and_init(gpos=(.5,.5,.5))
joints = w.getjoints()
frames = w.getframes()
bodies = w.getbodies()
arm_bodies = [bodies[n] for n in ['Arm', 'Forearm', 'Hand']]

#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import JointTask, CoMTask, LQPCoMTask
from LQPctrl.task_ctrl import KpCtrl
from arboris.homogeneousmatrix import rotz
tasks = []

goal = [-0.4, .2, 0]
tasks.append(JointTask(joints["Wrist"], KpCtrl(.5, 10.), [], 1., 0, True))
#tasks.append(CoMTask(arm_bodies, KpCtrl(goal, 10.), [], 1., 0, True))
# ...or...
tasks.append(LQPCoMTask(KpCtrl(goal, 10.), [], 1., 0, True))


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

from common import RecordCoMPosition
obs.append(RecordCoMPosition(arm_bodies))

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

import pylab as pl
pl.plot(obs[-1].get_record())
xlim = pl.xlim()
pl.plot(xlim, [-.4,-.4], 'r:')
pl.plot(xlim, [ .2, .2], 'r:')
pl.ylim([-.5, .5])
pl.ylabel("Position (m)")
pl.legend(['x','y','z'])
pl.xlabel("step")
pl.title("CoM Evolution")
pl.show()

