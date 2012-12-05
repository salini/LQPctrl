#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=15 june 2011

from common import create_icub_and_init, get_usual_observers
from arboris.robots import icub
from arboris.core import SubFrame, Observer
from arboris.shapes import Sphere, Cylinder
from arboris.homogeneousmatrix import transl

class ObsFrame(Observer):
    def __init__(self, f):
        self.f = f

    def init(self, world, timeline):
        pass
    def update(self, dt):
        print "F.pose: ", self.f.pose
    def finish(self):
        pass
#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_icub_and_init(gravity=True)

w.register(Sphere(SubFrame(w.ground, transl(-.16, -0.1, 0.5), name='obstacle'),radius=.05, name='obstacle') )
w.register(Sphere(SubFrame(w.ground, transl(-.16, -0.1, 0.63), name='obstacle'),radius=.05, name='obstacle2') )
#w.register(Cylinder(SubFrame(w.ground, transl(-.5, 0, 0.7), name='obstacle'),radius=.1, length=.2, name='obstacle') )

joints = w.getjoints()
frames = w.getframes()
bodies = w.getbodies()
shapes = w.getshapes()
consts = w.getconstraints()

icub_joints = [joints[n] for n in icub.get_joints_data()]
standing_pose = [j.gpos[0] for j in icub_joints]

#########################################
#                                       #
# CREATE TASKS, EVENTS & LQP controller #
#                                       #
#########################################
## TASKS
from LQPctrl.task import MultiJointTask, FrameTask, JointTask
from LQPctrl.task_ctrl import KpCtrl
from arboris.homogeneousmatrix import rotz
tasks = []

tasks.append(MultiJointTask(icub_joints, KpCtrl(standing_pose, 10), [], .01 , 0, True, "standing_pose"))
from numpy import eye
root_goal = eye(4)
root_goal[0:3,3] =  [0,0,.59]
tasks.append(JointTask(joints['root'], KpCtrl(root_goal, 10), [], 1 , 0, True, "root"))
spine_joints = [joints['torso_'+n] for n in ['pitch', 'roll', 'yaw']]
tasks.append(MultiJointTask(spine_joints, KpCtrl([0,0,0], 10), [], 1 , 0, True, "spine"))
#from arboris.homogeneousmatrix import rotz, rotx
#from numpy import pi
from numpy import array, cos, sin, pi
hand_goal = array([[-1,0 ,0 ,-0.2],
                   [0 ,-1 ,0,0],
                   [0 ,0,1 ,0.6],
                   [0 ,0 ,0 ,1]])
tasks.append(FrameTask(frames['l_hand_palm'], KpCtrl(hand_goal, 10), [], 1, 0, True, "hand")  )
## EVENTS
events = []


## LQP CONTROLLER
from LQPctrl.LQPctrl import LQPcontroller
gforcemax = icub.get_torque_limits()

data={}
data = {'collision shapes':[(shapes['l_hand'], shapes['obstacle']), (shapes['l_hand'], shapes['obstacle2'])]}
opt = {'avoidance horizon': 1.5}
sopt = {"show_progress":False}
lqpc = LQPcontroller(gforcemax, tasks=tasks, data=data, options=opt, solver_options=sopt)
w.register(lqpc)


############################
#                          #
# SET OBSERVERS & SIMULATE #
#                          #
############################
obs = get_usual_observers(w)
obs.append(ObsFrame(frames['l_hand_palm']))
from common import RecordCoMPosition

## SIMULATE
from numpy import arange
from arboris.core import simulate
simulate(w, arange(0,5.,0.01), obs)


###########
#         #
# RESULTS #
#         #
###########
print("end of the simulation")


