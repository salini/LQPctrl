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

dt = 0.01


#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import FrameTask
from LQPctrl.task_ctrl import KpTrajCtrl
from arboris.homogeneousmatrix import rotz
from numpy import pi, tile, cos, sin, zeros, arange
tasks = []

amp, T, phi = .5, 4., pi/2
omega = 2*pi/T
t = arange(0, 1.5*T, dt)
x, vx, ax = amp*cos(t*omega + phi), -omega*amp*sin(t*omega + phi), -(omega**2)*amp*cos(t*omega + phi)
y, vy, ay = amp*sin(t*omega + phi),  omega*amp*cos(t*omega + phi), -(omega**2)*amp*sin(t*omega + phi)
pos, vel, acc = tile(rotz(pi/8), (len(t),1,1)), tile(zeros(6), (len(t),1)), tile(zeros(6), (len(t),1))
pos[:,0,3], pos[:,1,3] = x , y
vel[:,3], vel[:,4]     = vx, vy
acc[:,3], acc[:,4]     = ax, ay
goal = [pos, vel, acc]
tasks.append(FrameTask(frames["EndEffector"], KpTrajCtrl(goal, 20), [], 1., 0, True))


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
simulate(w, arange(0,8,dt), obs)


###########
#         #
# RESULTS #
#         #
###########
print("end of the simulation")

import pylab as pl
res = obs[-1].get_record()
pl.plot(res)
xlim = pl.xlim()
pl.plot(x[:len(res)], 'b:', lw=2)
pl.plot(y[:len(res)], 'g:', lw=2)
pl.ylabel("Position (m)")
pl.legend(['x','y','z', 'xdes', 'ydes'])
pl.xlabel("step")
pl.title("EndEffector Evolution")
pl.show()

