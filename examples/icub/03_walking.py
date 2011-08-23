#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 june 2011

from common import create_icub_and_init, get_usual_observers, print_lqp_perf
from arboris.robots import icub
from arboris.homogeneousmatrix import rotx
from numpy import pi, eye, zeros



#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_icub_and_init(gravity=True)
joints = w.getjoints()
frames = w.getframes()
bodies = w.getbodies()
frames = w.getframes()
const  = w.getconstraints()

icub_joints = [joints[n] for n in icub.get_joints_data()]
icub_bodies = [bodies[n] for n in icub.get_bodies_data()]
standing_pose = [j.gpos[0] for j in icub_joints]
back_joints = [joints[n] for n in ['torso_pitch', 'torso_roll', 'torso_yaw', 'head_pitch', 'head_roll', 'head_yaw']]
arms_joints = [joints[n] for n in ['l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow_pitch', 'l_elbow_yaw', 'l_wrist_roll', 'l_wrist_pitch']+
               ['r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow_pitch', 'r_elbow_yaw', 'r_wrist_roll', 'r_wrist_pitch']]

arms_goal = [j.gpos[0] for j in arms_joints]
l_const = [c for c in const if c.name[0:2] == 'lf']
r_const = [c for c in const if c.name[0:2] == 'rf']

dt = 0.01

#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import MultiJointTask, JointTask
from LQPctrl.task_ctrl import KpCtrl
from LQPctrl.walk import WalkingCtrl, WalkingTask
tasks = []


tasks.append(MultiJointTask(icub_joints, KpCtrl(standing_pose, 10), [], 1e-2 , 0, True, "standing_pose"))


goal = {"action": "goto", "pos": [-0.2,0]}
zmp = {'QonR':1e-6, 'horizon':1.7, 'dt':dt, 'cdof':[0,1]}
feet = {"Kp":150, "Kd":None, "weight":1e+1,
        "l_frame":frames['l_sole'], "r_frame":frames['r_sole'], "R0":rotx(pi/2.), "l_const":l_const, "r_const":r_const}
#step={"length":.1, "side":.05, "height":.01, "time": .8, "ratio":.7, "start": "right"}
step={"length":.05, "side":.05, "height":.01, "time": .8, "ratio":.7, "start": "left"}
wctrl = WalkingCtrl(goal, zmp, feet, step)
tasks.append(WalkingTask(icub_bodies, wctrl, [], 1., 0, True, "walk"))


## LQP CONTROLLER
from LQPctrl.LQPctrl import LQPcontroller
gforcemax = icub.get_torque_limits()

#opt = {'base weights'   : (1e-8, 1e-8, 1e-8)} #, 'formalism':'chi'
#sopt = {'show_progress':False,
#        'abstol'        : 1e-11,
#        'reltol'        : 1e-10,
#        'feastol'       : 1e-11,}
#lqpc = LQPcontroller(gforcemax, tasks=tasks, options=opt, solver_options=sopt)
lqpc = LQPcontroller(gforcemax, tasks=tasks)
w.register(lqpc)


############################
#                          #
# SET OBSERVERS & SIMULATE #
#                          #
############################
obs = get_usual_observers(w)

from common import RecordCoMPosition, RecordZMPPosition
obs.append(RecordCoMPosition(icub_bodies))
obs.append(RecordZMPPosition(icub_bodies))

## SIMULATE
from numpy import arange
from arboris.core import simulate
simulate(w, arange(0,6.,dt), obs)



###########
#         #
# RESULTS #
#         #
###########
print("end of the simulation")
print_lqp_perf(lqpc)

from numpy import array
import pylab as pl
pl.figure()
com = array(obs[-2].get_record())
pl.plot(com[:,[0,1]])
pl.ylabel("Position (m)")
pl.legend(['x','y'])
pl.xlabel("step")
pl.title("CoM Evolution")

pl.figure()
zmp = array(obs[-1].get_record())
pl.plot(zmp[:,[0,1]])
pl.ylabel("Position (m)")
pl.legend(['x','y'])
pl.xlabel("step")
pl.title("ZMP Evolution")

pl.show()

