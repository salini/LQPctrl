#coding=utf-8
#author=Joseph Salini
#date=16 june 2011

from task import MultiTask, CoMTask, LQPCoMTask, FrameTask
from task_ctrl import Ctrl, ZMPCtrl, KpTrajCtrl

from numpy import dot, arctan2, asarray, array, linspace, ones, \
                  sin, cos, vstack, arange, pi, zeros, tile, eye
from numpy.linalg import norm

from arboris.homogeneousmatrix import rotx, roty, rotz

from scipy.interpolate import piecewise_polynomial_interpolate as ppi

################################################################################
#
# Miscalleneous Functions for Walking
#
################################################################################

def traj2zmppoints(comtraj, step_length, step_side, \
                   left_start, right_start, start_foot='left'):
    """Generate a set of points to locate the feet position around a trajectory
    of the Center of Mass.

    :param comtraj: list of 3 parameters: [x_traj, y_traj, angular_traj]

    :param step_length: the distance done with one step in meter
    :param step_side: the distance between the feet and the CoM trajectory

    :param left_start: left foot pos [x_pos, y_pos, angular_pos]
    :param right_start: right foot pos [x_pos, y_pos, angular_pos]

    :param start_foot: 'left'/'l' or 'right'/'r'
    :type start_foot: string

    :return: a list of points which represent the feet location on floor

    """
    left_start = array(left_start)
    right_start = array(right_start)
    point = []

    if   start_foot == 'left' :
        point.extend([left_start, right_start])
    elif start_foot == 'right':
        point.extend([right_start, left_start])
    else: raise ValueError
    next_foot = start_foot

    sum_distance = 0.
    for i in arange(len(comtraj)-1):
        sum_distance += norm(comtraj[i+1][0:2]-comtraj[i][0:2])

        if sum_distance > step_length:
            angle = comtraj[i][2]
            ecart = step_side*array([-sin(angle), cos(angle), 0])
            if next_foot is 'right':
                ecart = -ecart
            point.append(comtraj[i] + ecart)
            sum_distance = 0.
            next_foot = 'right' if next_foot == 'left' else 'left'

    # just to get the 2 last footsteps
    angle = comtraj[-1][2]
    ecart = step_side*array([-sin(angle), cos(angle), 0])
    if next_foot == 'left':
        point.extend([comtraj[-1] + ecart, comtraj[-1] - ecart])
    else:
        point.extend([comtraj[-1] - ecart, comtraj[-1] + ecart])
    return point



def zmppoints2zmptraj(point, step_time, dt):
    """Get the Zero Moment Point trajectory from feet location.

    :param point: the list of the feet location
    :param step_time: the time between 2 steps
    :param dt: dt of simulation

    :return: the ZMP traj [x_traj, y_traj]

    """
    gab2 = ones((round(step_time/dt/2), 1))
    gab  = ones((round(step_time/dt),   1))

    start = dot(gab2, point[0][0:2].reshape(1, 2))
    mid   = [dot(gab, p[0:2].reshape(1, 2)) for p in point[1:-1]]
    end   = (point[-2][0:2] + point[-1][0:2])/2.
    traj  = vstack( [start]+mid+[end] )

    return traj



def zmppoints2foottraj(point, step_time, ratio, step_height, dt, cdof, R0):
    """Compute the trajectory of the feet.

    :param point: the list of the feet location
    :param step_time: the time between 2 steps
    :param ratio: ratio between sigle support phase time and step_time
    :param step_height: the max distance between the foot and the floor
    :param dt: dt of simulation
    :param Hfloor: the transformation matrix of the floor

    :return: a list with all step trajectories [(pos_i, vel_i, acc_i)]

    """
    def get_bounded_angles(p0, p1):
        #WARNING: do this trick to get the shortest path:
        a0, a1 = (p0[2])%(2*pi), (p1[2])%(2*pi)
        diff = abs(a1 - a0)
        if   abs(a1+2*pi - a0) <diff:
            a1 += 2*pi
        elif abs(a1-2*pi - a0) <diff:
            a1 -= 2*pi
        return a0, a1

    foot_traj = []
    if   0 not in cdof:
        up = 0
        operation = rotx
    elif 1 not in cdof:
        up = 1
        operation = roty
    elif 2 not in cdof:
        up = 2
        operation = rotz

    xout      = arange(0, step_time*ratio+dt, dt)
    xin, xin2 = [0, step_time*ratio], [0, step_time*ratio/2, step_time*ratio]
    yup       = [[0, 0, 0], [step_height, 0], [0, 0, 0]]

    for i in arange(len(point)-2):
        yin0   = [[point[i][0], 0, 0], [point[i+2][0], 0, 0]]
        yin1   = [[point[i][1], 0, 0], [point[i+2][1], 0, 0]]
        a_start, a_end = get_bounded_angles(point[i], point[i+2])
        yangle = [[a_start, 0, 0], [a_end, 0, 0]]
        data = [(cdof[0], xin, yin0), \
                (cdof[1], xin, yin1), \
                (up,xin2, yup), \
                ('angle', xin, yangle)]
        res = {}
        for c, xx, yy in data:
            res[c] = (ppi(xx, yy, xout), \
                      ppi(xx, yy, xout, der=1),
                      ppi(xx, yy, xout, der=2))

        ## save traj
#        pos = zeros((len(xout), 4, 4))
        pos = tile(eye(4), (len(xout), 1, 1))
        vel = zeros((len(xout), 6))
        acc = zeros((len(xout), 6))

        for j in arange(len(xout)):
            pos[j, 0:3, 0:3] = dot(operation(res['angle'][0][j])[0:3, 0:3], \
                                  R0[0:3, 0:3] )
        vel[:, up] = res['angle'][1]
        acc[:, up] = res['angle'][2]

        for j in arange(3):
            pos[:, j, 3] = res[j][0]
            vel[:, 3+j]  = res[j][1]
            acc[:, 3+j]  = res[j][2]

        foot_traj.append( [pos, vel, acc] )
    return foot_traj








################################################################################
#
# Walking Ctrl
#
################################################################################
class WalkingCtrl(Ctrl):
    def __init__(self, goal, zmp_args, feet, step):
        Ctrl.__init__(self)
        self._goal = goal
        self.feet  = feet
        self.step  = step

        self.zmp_ctrl = ZMPCtrl([], **zmp_args)

        Kp, Kd, weight = feet['Kp'], feet['Kd'], feet['weight']
        self.l_foot_ctrl = KpTrajCtrl([None, None, None], Kp, Kd)
        self.r_foot_ctrl = KpTrajCtrl([None, None, None], Kp, Kd)
        self.l_foot_task = FrameTask(feet['l_frame'], self.l_foot_ctrl, \
                                     [], weight, 0, False, "l_foot")
        self.r_foot_task = FrameTask(feet['r_frame'], self.r_foot_ctrl, \
                                     [], weight, 0, False, "r_foot")
        
        self.cdof = zmp_args['cdof']
        self.dt   = zmp_args['dt']
        self._R0  = feet["R0"][0:3, 0:3]
        self._iR0 = self._R0.T

        self.events = []
        self._num_step = 0
        self._sequence = []
        self._prev_foot = None
        self._next_foot = None
        self._foot_traj = None


    def init(self, world, LQP_ctrl):
        self.world = world
        self.LQP_ctrl = LQP_ctrl
        self.set_goal(self._goal)


    def set_goal(self, new_goal):
        assert(isinstance(new_goal, dict))
        action = new_goal["action"]
        if action == 'idle':
            mid_feet = self._get_center_of_feet()
            self.zmp_ctrl.set_goal([mid_feet])
        if action == 'goto':
            start = self._get_center_of_feet()
            end   = asarray(new_goal["pos"])
            vect = (end - start)
            if "angle" in new_goal:
                angle = new_goal["angle"]
            else:
                angle = arctan2(vect[1], vect[0])
            traj = array([linspace(start[0], end[0], 10000), \
                          linspace(start[1], end[1], 10000), \
                          angle*ones(10000)]).T
            s = self.step
            l_start, r_start = self._get_lf_pose(), self._get_rf_pose()
            points  = traj2zmppoints(traj, s["length"], s["side"], \
                                     l_start, r_start, s["start"])
            zmp_ref = zmppoints2zmptraj(points, s["time"], self.dt)
            self.zmp_ctrl.set_goal(zmp_ref)
            ftraj = zmppoints2foottraj(points, s["time"], \
                                       s["ratio"], s["height"], \
                                       self.dt, self.cdof, self._R0)
            self._sequence = self.world.current_time + \
                             (arange(len(ftraj)+1) + .5)*self.step["time"]
            self._foot_traj = ftraj
            self._num_step = 0
            self._next_foot = self.step['start']
            self._prev_foot = 'right' if \
                              self.step['start'] == 'left' else 'left'

    def update(self, rstate, dt):
        if len(self._sequence) and self._num_step < len(self._sequence):
            t = self.world.current_time
            sqt = self._sequence[self._num_step]
            r = self.step['time']*(1 - self.step['ratio'])/2.
            if t >= sqt - r:
                #print "desactivate FOOT", self._prev_foot
                self._end_foot(self._prev_foot)
            if t >= sqt + r:
                print "ACTIVATE FOOT", self._next_foot
                if self._num_step < len(self._foot_traj):
                    self._start_foot(self._next_foot, \
                                     self._foot_traj[self._num_step])
                self._prepare_to_next_foot()


    def _end_foot(self, foot):
        if   foot == 'left' :
            task = self.l_foot_task
            const = self.feet['l_const']
        elif foot == 'right':
            task = self.r_foot_task
            const = self.feet['r_const']
        for c in const:
            self.LQP_ctrl.is_enabled[c] = True
        task._is_active = False

    def _start_foot(self, foot, traj):
        if   foot == 'left' :
            task = self.l_foot_task
            const = self.feet['l_const']
        elif foot == 'right':
            task = self.r_foot_task
            const = self.feet['r_const']
        for c in const:
            self.LQP_ctrl.is_enabled[c] = False
        task._is_active = True
        task.ctrl.set_goal(traj)

    def _prepare_to_next_foot(self):
        tmp = self._next_foot
        self._next_foot = self._prev_foot
        self._prev_foot = tmp
        self._num_step += 1



    def _get_foot_pose(self, pose):
        pos   = pose[self.cdof, 3]
        H0a0  = dot(pose[0:3, 0:3], self._iR0)
        angle = arctan2(H0a0[self.cdof[1], self.cdof[0]], \
                        H0a0[self.cdof[0], self.cdof[0]])
        return array([pos[0], pos[1], angle])

    def _get_lf_pose(self):
        return self._get_foot_pose(self.feet["l_frame"].pose)

    def _get_rf_pose(self):
        return self._get_foot_pose(self.feet["r_frame"].pose)

    def _get_center_of_feet(self):
        lf = self._get_lf_pose()[0:2]
        rf = self._get_rf_pose()[0:2]
        return (lf+rf)/2.


################################################################################
#
# Walking Task
#
################################################################################
class WalkingTask(MultiTask):
    def __init__(self, bodies, ctrl, *args, **kargs):
        MultiTask.__init__(self, *args, **kargs)
        self._bodies = bodies
        self._ctrl = ctrl

        com_args = [ctrl.zmp_ctrl, ctrl.cdof, 1, 0, True, "CoM"]
        if len(bodies):
            self._subtask.append( CoMTask(bodies, *com_args) )
        else:
            self._subtask.append( LQPCoMTask(*com_args) )
        self._subtask.append( ctrl.l_foot_task )
        self._subtask.append( ctrl.r_foot_task )


    def init(self, world, LQP_ctrl):
        MultiTask.init(self, world, LQP_ctrl)
        self._ctrl.init(world, LQP_ctrl)


    def update(self, rstate, dt, _rec_performance):
        MultiTask.update(self, rstate, dt, _rec_performance)
        self._ctrl.update(rstate, dt)


