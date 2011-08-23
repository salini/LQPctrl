#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

from common import create_3r_and_init, get_usual_observers, print_lqp_perf

from arboris.core import SubFrame
from arboris.shapes import Sphere, Cylinder
from arboris.homogeneousmatrix import transl
from arboris.constraints import SoftFingerContact


#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_3r_and_init(gpos=(.5,.5,.5))
frames = w.getframes()

w.register(Sphere(frames['EndEffector'],radius=.02, name='ee') )
w.register(Sphere(SubFrame(w.ground, transl(-.8, .2, 0.), name='obstacle'),radius=.1, name='obstacle') )
w.register(Sphere(SubFrame(w.ground, transl(-.55, .2, 0.), name='obstacle2'),radius=.1, name='obstacle2') )

shapes = w.getshapes()

#w.register(SoftFingerContact((shapes['ee'], shapes['obstacle']), .01))



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
goal = rotz(3*pi/4)
goal[0:3,3] = [-.75, -.0, 0]
tasks.append(FrameTask(frames["EndEffector"], KpCtrl(goal, 2), [], 1., 0, True))


## EVENTS
events = []


## LQP CONTROLLER
from LQPctrl.LQPctrl import LQPcontroller
gforcemax = {"Shoulder":10,"Elbow":5,"Wrist":2}

data={}
data = {'collision shapes':[(shapes['ee'], shapes['obstacle']), (shapes['ee'], shapes['obstacle2'])]}
opt = {'avoidance horizon': 1.}
sopt = {"show_progress":False}
lqpc = LQPcontroller(gforcemax, tasks=tasks, data=data, options=opt, solver_options=sopt)
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
simulate(w, arange(0,10,0.01), obs)


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
pl.plot(xlim, [goal[0,3], goal[0,3]], 'r:')
pl.plot(xlim, [goal[1,3], goal[1,3]], 'r:')
pl.ylim([-1., 1.])
pl.ylabel("Position (m)")
pl.legend(['x','y','z'])
pl.xlabel("step")
pl.title("EndEffector Evolution")
pl.show()

