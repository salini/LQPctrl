#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=15 june 2011

from common import create_icub_and_init, get_usual_observers
from arboris.robots import icub

#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_icub_and_init(gravity=True)
joints = w.getjoints()
frames = w.getframes()
bodies = w.getbodies()
consts = w.getconstraints()

icub_joints = [joints[n] for n in icub.get_joints_data()]
standing_pose = [j.gpos[0] for j in icub_joints]

#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import MultiJointTask
from LQPctrl.task_ctrl import KpCtrl
from arboris.homogeneousmatrix import rotz
tasks = []

tasks.append(MultiJointTask(icub_joints, KpCtrl(standing_pose, 10), [], 1 , 0, True, "standing_pose"))

## EVENTS
events = []


## LQP CONTROLLER
from LQPctrl.LQPctrl import LQPcontroller
gforcemax = icub.get_torque_limits()

lqpc = LQPcontroller(gforcemax, tasks=tasks)
w.register(lqpc)


############################
#                          #
# SET OBSERVERS & SIMULATE #
#                          #
############################
obs = get_usual_observers(w)

from common import RecordCoMPosition

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


