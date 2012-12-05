#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=15 june 2011

from common import create_icub_and_init, get_usual_observers
from arboris.robots import icub
from arboris.core import NamedObjectsList

from numpy import pi, array, zeros
#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_icub_and_init(chair=True, gravity=True)

joints = w.getjoints()
frames = w.getframes()
bodies = w.getbodies()


icub_joints = [joints[n] for n in icub.get_joints_data()]
icub_bodies = [bodies[n] for n in icub.get_bodies_data()]
standing_pose = [j.gpos[0] for j in icub_joints]

joints['l_knee'].gpos[:] = -pi/2
joints['r_knee'].gpos[:] = -pi/2
joints['l_ankle_pitch'].gpos[:] = -pi/10
joints['r_ankle_pitch'].gpos[:] = -pi/10
joints['l_hip_pitch'].gpos[:] = pi/2 - pi/10
joints['r_hip_pitch'].gpos[:] = pi/2 - pi/10
joints['root'].gpos[0:3,3] = array([0,0,.597]) - [-0.2, 0, 0.2231] + [-0.01916-0.012, 0, 0.0587]
sitting_pose = [j.gpos[0] for j in icub_joints]

const = w.getconstraints()
w.update_dynamic()

spine_joints = [joints['torso_'+n] for n in ['pitch', 'roll', 'yaw']] + \
              [joints['head_' +n] for n in ['pitch', 'roll', 'yaw']]

#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import MultiJointTask, ForceTask, FrameTask, CoMTask
from LQPctrl.task_ctrl import KpCtrl, ValueCtrl, KpTrajCtrl
tasks = NamedObjectsList([])

tasks.append(MultiJointTask(icub_joints, KpCtrl(sitting_pose, 10), [], 1e-3 , 0, True, "sitting_pose"))
tasks.append(MultiJointTask(spine_joints, KpCtrl(zeros(len(spine_joints)), 10), [], 1e-2 , 0, True, "spine"))

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

from common import RecordCoMPosition, RecordGforce
obs.append(RecordGforce(lqpc))


## SIMULATE
from numpy import arange
from arboris.core import simulate
simulate(w, arange(0,1.5,0.01), obs)


###########
#         #
# RESULTS #
#         #
###########
print("end of the simulation")


import pylab as pl
pl.plot(obs[-1].get_record())
xlim = pl.xlim()
pl.ylabel("Tau (N.m)")
pl.xlabel("step")
pl.title("Gforce Evolution")

pl.show()

