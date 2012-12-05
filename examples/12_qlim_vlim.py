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
w = create_3r_and_init(gpos=(.7,.6,.5))
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

qlim = {"Shoulder":(.2,1.1),"Elbow":(.3,1.1),"Wrist":(.4,1.1)}
vlim = {"Shoulder":.35,"Elbow":.2,"Wrist":.15}
opt  = {'pos horizon':.2, 'vel horizon': .1}
lqpc = LQPcontroller(gforcemax, qlim=qlim, vlim=vlim, tasks=tasks, options=opt)
w.register(lqpc)


############################
#                          #
# SET OBSERVERS & SIMULATE #
#                          #
############################
obs = get_usual_observers(w)

from common import RecordJointPosition, RecordJointVelocity
obs.append(RecordJointPosition(joints))
obs.append(RecordJointVelocity(joints))

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
pl.figure()
pl.plot(obs[-2].get_record())
xlim = pl.xlim()
for v in [.2,.3,.4]: pl.plot(xlim, [v,v], 'r:')
pl.plot(xlim, [.1,.1], 'k-.')
pl.ylim([0, .8])
pl.ylabel("position (rad)")
pl.xlabel("step")
pl.title("Arm Joints Evolution")
pl.legend(["Shoulder","Elbow","Wrist"])

pl.figure()
pl.plot(obs[-1].get_record())
for v in [-.35,-.2,-.15]: pl.plot(xlim, [v,v], 'r:')
pl.ylim([-.4, .1])
pl.ylabel("velocity (rad/s)")
pl.xlabel("step")
pl.title("Arm Joints Evolution")
pl.legend(["Shoulder","Elbow","Wrist"])

pl.show()

