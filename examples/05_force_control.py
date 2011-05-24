#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

from common import create_3r_and_init, add_plane_and_point_on_arm, get_usual_observers, print_lqp_perf
from numpy import pi, array

#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_3r_and_init(gpos=(pi/2.,pi/6.,pi/3.), gravity=False)
add_plane_and_point_on_arm(w, (0,1,0,-.4))
frames = w.getframes()
shapes = w.getshapes()
const  = w.getconstraints()



#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import ForceTask, FrameTask
from LQPctrl.task_ctrl import ValueCtrl, KpCtrl
from arboris.homogeneousmatrix import rotz, transl
from numpy import pi
tasks = []
goal = rotz(pi/8)
goal[0:3,3] = [-0.4, .5, 0]
goal = transl(-.8,-.5,0)
tasks.append(ForceTask( const["const"], ValueCtrl([0, 0.001,.01]) , [], 1., 0, True))



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

from common import RecordWrench
obs.append(RecordWrench(const["const"]))

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
pl.ylim([0,.015])
pl.ylabel("Force (N)")
pl.legend(['tau_z', 'x','y','z'])
pl.xlabel("step")
pl.title("EndEffector Evolution")
pl.show()

