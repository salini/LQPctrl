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
tasks.append(MultiJointTask(icub_joints, KpCtrl(standing_pose, 10), [], 0. , 0, True, "standing_pose"))
tasks.append(MultiJointTask(spine_joints, KpCtrl(zeros(len(spine_joints)), 10), [], 1e-2 , 0, True, "spine"))
tasks.append(ForceTask( const["l_gluteal"], ValueCtrl([0, 0, 0]) , [], 0., 0, True, "l_gluteal_force"))
tasks.append(ForceTask( const["r_gluteal"], ValueCtrl([0, 0, 0]) , [], 0., 0, True, "r_gluteal_force"))
tasks.append(FrameTask( frames["l_hip_2"], KpTrajCtrl([None,None,zeros((1,6))], 10), [3,4,5], 1., 0, True, "l_gluteal_acc"))
tasks.append(FrameTask( frames["r_hip_2"], KpTrajCtrl([None,None,zeros((1,6))], 10), [3,4,5], 1., 0, True, "r_gluteal_acc"))
tasks.append(CoMTask(icub_bodies, KpCtrl([0, 0, 0], 10), [0, 1], 0., 0, name='com'))

## EVENTS
from LQPctrl.event import Event, AtT, Prtr, SetF, IfF, ChW, InCH, CAct, DelayF
events = []
tt=.25
feet_frames = [frames[n] for n in ['lf1', 'lf2', 'lf3', 'lf4', 'rf1', 'rf2', 'rf3', 'rf4']]
events.append(Event(AtT(.0),
                    SetF("start lifting", False)))
events.append(Event(AtT(.1),
                    SetF("avance com", True)))
events.append(Event(InCH(feet_frames, 'CoM', [0,1]),
                    [SetF("in CH", True)] ))
events.append(Event(IfF("avance com", True),
                    [Prtr("avance com"), ChW(tasks['com'], 1., tt, 1e-3), SetF("avance com", False)]))
events.append(Event([IfF("start lifting", False), IfF("in CH", True)],
                    [ChW(tasks['l_gluteal_force'], 1e-5, tt), ChW(tasks['r_gluteal_force'], 1e-5, tt), Prtr("start lifiting"),
                     DelayF("can del const", True, tt),
                     SetF("start lifting", True)]))
events.append(Event(IfF("can del const", True),
                    [Prtr("CAN DEL CONST"),CAct(const["l_gluteal"], False, True), CAct(const["r_gluteal"], False, True),
                     ChW(tasks['standing_pose'], 1e-3, tt), 
                     ChW(tasks['l_gluteal_acc'], 0, tt), ChW(tasks['r_gluteal_acc'], 0, tt), DelayF("can del sitting", True, tt),
                     SetF("can del const", False)]))
events.append(Event(IfF("can del sitting", True),
                    [ChW(tasks['sitting_pose'], 0., tt), SetF("can del sitting", False)]))


## LQP CONTROLLER
from LQPctrl.LQPctrl import LQPcontroller
gforcemax = icub.get_torque_limits()
dgforcemax = dict([(k,2.*v) for k,v in gforcemax.items()])
opt = {'base weights' : (1e-7, 1e-7, 1e-7)}
sopt = {'show_progress':False,
        'abstol'        : 1e-12,
        'reltol'        : 1e-12,
        'feastol'       : 1e-12,}
lqpc = LQPcontroller(gforcemax, dgforcemax, tasks=tasks, events=events, options = opt, solver_options=sopt)
w.register(lqpc)


############################
#                          #
# SET OBSERVERS & SIMULATE #
#                          #
############################
obs = get_usual_observers(w)

from common import RecordCoMPosition, RecordGforce
obs.append(RecordGforce(lqpc))
obs.append(RecordCoMPosition(icub_bodies))


## SIMULATE
from numpy import arange
from arboris.core import simulate
simulate(w, arange(0,3.,0.01), obs)


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
pl.ylabel("Tau (N.m)")
pl.xlabel("step")
pl.title("Gforce Evolution")

pl.figure()
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

