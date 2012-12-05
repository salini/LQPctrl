#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=15 june 2011

from common import create_icub_and_init, get_usual_observers
from arboris.robots import icub
from arboris.core import SubFrame, Controller
from arboris.shapes import Sphere, Cylinder, Box
from arboris.homogeneousmatrix import transl
from numpy import eye, sqrt, zeros, sin, pi

def add_table(world, hl=(.05, .05, .05), init_pos= eye(4), mass=1., name='wall'):
    """ Add a simple box to the world
    """
    from arboris.massmatrix import box
    from arboris.core import Body
    from arboris.joints import TzJoint
    ## Create the body
    M = box(hl, mass)
    bbox = Body(mass=M, name=name)
    world.register(Box(bbox, hl, name))

    ## Link the body to the world
    world.add_link(SubFrame(world.ground, init_pos), TzJoint(name=name+'_root'), bbox)

    world.init()
    
    
class KpTable(Controller):
    def __init__(self, body, amp, period, kp):
        self.body = body
        self.parentjoint = body.parentjoint
        self.amp = amp
        self.period = period
        self.kp = kp
        self.kd = 2*sqrt(kp)
        self.mass = body.mass[5,5]

    def init(self, world):
        self.world = world

    def update(self, dt):
        gforce = zeros(self.world.ndof)
        impedance = zeros((self.world.ndof, self.world.ndof))
        t = self.world._current_time

        posdes = self.amp*sin(t*2.*pi/self.period)
#        posdes = self.amp*sin(t*2.*pi/self.period + pi/2) - self.amp
        pos = self.parentjoint.gpos
        vel = self.parentjoint.gvel
        force = self.kp*(posdes-pos) - self.kd*vel
        gforce[self.parentjoint.dof] = force
        gforce[self.parentjoint.dof] += self.mass*9.81
        return gforce, impedance



#################################
#                               #
# CREATE WORLD & INITIALIZATION #
#                               #
#################################
w = create_icub_and_init(gravity=True)

add_table(w, hl=(.05,.2,.02), init_pos=transl(-.3,0,.6), mass=10., name='wall')

joints = w.getjoints()
frames = w.getframes()
bodies = w.getbodies()
shapes = w.getshapes()
consts = w.getconstraints()

##POS of the arms, before or after the rec of standing_pos?
joints['l_shoulder_pitch'].gpos[:] = -0.65695599
joints['l_shoulder_roll'].gpos[:] = 0.58153898
joints['l_shoulder_yaw'].gpos[:] = 0.68249881
joints['l_elbow_pitch'].gpos[:] = 1.27058836
joints['l_elbow_yaw'].gpos[:] = 0.85522174
joints['l_wrist_roll'].gpos[:] = 0.15702068
joints['l_wrist_pitch'].gpos[:] = 0.41882797

joints['r_shoulder_pitch'].gpos[:] = -0.65695599
joints['r_shoulder_roll'].gpos[:] = 0.58153898
joints['r_shoulder_yaw'].gpos[:] = 0.68249881
joints['r_elbow_pitch'].gpos[:] = 1.27058836
joints['r_elbow_yaw'].gpos[:] = 0.85522174
joints['r_wrist_roll'].gpos[:] = 0.15702068
joints['r_wrist_pitch'].gpos[:] = 0.41882797

w.update_dynamic()

icub_joints = [joints[n] for n in icub.get_joints_data()]
icub_bodies = [bodies[n] for n in icub.get_bodies_data()]
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

## EVENTS
events = []


## LQP CONTROLLER
from LQPctrl.LQPctrl import LQPcontroller
gforcemax = icub.get_torque_limits()

data={'bodies':icub_bodies}
data.update({'collision shapes':[(shapes['l_hand_palm'], shapes['wall']), (shapes['r_hand_palm'], shapes['wall'])]})
opt = {'avoidance horizon': .1}
sopt = {"show_progress":False}
lqpc = LQPcontroller(gforcemax, tasks=tasks, data=data, options=opt, solver_options=sopt)
w.register(lqpc)

w.register(KpTable(bodies['wall'], 0.05, 2., 100.))
############################
#                          #
# SET OBSERVERS & SIMULATE #
#                          #
############################
obs = get_usual_observers(w)

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


