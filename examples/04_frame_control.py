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
frames = w.getframes()


#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import FrameTask
from LQPctrl.task_ctrl import KpCtrl
from arboris.homogeneousmatrix import rotz
from numpy import pi
tasks = []
goal = rotz(pi/8)
goal[0:3,3] = [-0.4, .5, 0]
tasks.append(FrameTask(frames["EndEffector"], KpCtrl(goal, 20), [], 1., 0, True))


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

from common import RecordFramePosition
obs.append(RecordFramePosition(frames["EndEffector"]))

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
pl.plot(xlim, [ .5, .5], 'r:')
pl.ylim([-1., 1.])
pl.ylabel("Position (m)")
pl.legend(['x','y','z'])
pl.xlabel("step")
pl.title("EndEffector Evolution")
pl.show()

