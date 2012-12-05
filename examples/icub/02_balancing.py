#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=15 june 2011

from common import create_icub_and_init, get_usual_observers
from arboris.robots import icub

def get_zmp_traj(_type, opt):
    from numpy import sin, arange, pi, hstack, zeros, ones, tile
    if _type == 'constant':
        return [[opt['x'], opt['y']]]
    elif _type == 'sin' or _type == 'square':
        T, dt, amp, t0, tend  = opt['T'], opt['dt'], opt['amp'], opt['t0'], opt['tend']
        t = arange(0, (tend - t0), dt)
        if _type == 'sin':
            y = amp*sin(t*2*pi/T)
        elif _type == 'square':
            y = tile( hstack(( amp*ones(int(T/dt/2.)), -amp*ones(int(T/dt/2.)) )), int((tend - t0)/T + 1))[:len(t)]
        y = y = hstack((zeros(int(t0/dt)), y ))
        zmp_traj = zeros((len(y),2))
        zmp_traj[:,1] = y
        return zmp_traj



#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_icub_and_init(gravity=True)
joints = w.getjoints()
frames = w.getframes()
bodies = w.getbodies()

icub_joints = [joints[n] for n in icub.get_joints_data()]
icub_bodies = [bodies[n] for n in icub.get_bodies_data()]
standing_pose = [j.gpos[0] for j in icub_joints]

dt = 0.01

#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import MultiJointTask, CoMTask
from LQPctrl.task_ctrl import KpCtrl, ZMPCtrl
from arboris.homogeneousmatrix import rotz
tasks = []

tasks.append(MultiJointTask(icub_joints, KpCtrl(standing_pose, 10), [], 1e-4 , 0, True, "standing_pose"))

#zmp_traj = get_zmp_traj('constant', opt={'x':.02,'y':.01})
#zmp_traj = get_zmp_traj('sin', opt={'T':1,'dt':dt,'amp':.02, 't0':.5,'tend':4.})
zmp_traj = get_zmp_traj('square', opt={'T':1,'dt':dt,'amp':.02, 't0':.5,'tend':4.})

zmp_ctrl = ZMPCtrl(zmp_traj, QonR=1e-6, horizon=1.7, dt=dt, cdof=[0,1])
tasks.append(CoMTask(icub_bodies, zmp_ctrl, [0,1], 1, 0, True, "balance"))


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
pl.plot(zmp_traj[:len(zmp)], ':r')
pl.ylabel("Position (m)")
pl.legend(['x','y'])
pl.xlabel("step")
pl.title("ZMP Evolution")

pl.show()

