#coding=utf-8
#author=Joseph Salini
#date=21 april 2011


from abc import ABCMeta, abstractmethod
from arboris.core import Joint, Frame
from arboris.homogeneousmatrix import ishomogeneousmatrix
from arboris.controllers import WeightController

from numpy import zeros, array, asarray, dot, arange, ones, vstack, eye, sqrt
from numpy.linalg import inv

from misc import quatpos

def diff(val1, val2):
    """
    """
    if ishomogeneousmatrix(asarray(val1)):
        v1 = quatpos(val1)
    else:
        v1 = asarray(val1)
    if ishomogeneousmatrix(asarray(val2)):
        v2 = quatpos(val2)
    else:
        v2 = asarray(val2)
    return v1 - v2


def get_quadratic_cmd(Px, Pu, QonR, h, x_hat, z_ref):
    cmd_traj = - dot( inv(dot(Pu.T, Pu) + QonR*eye(h)), \
                      dot(Pu.T, dot(Px, x_hat) - z_ref) )
    return cmd_traj


class Ctrl:
    """
    """
    __metaclass__ = ABCMeta
    
    def __init__(self):
        self._error = 0.
        self._goal = None
        self.world = None
        self.LQP_ctrl = None

    def init(self, world, LQP_ctrl):
        pass

    def set_goal(self, new_goal):
        self._goal = new_goal

    @property
    def error(self):
        return self._error

################################################################################
#
# dTwist Controllers
#
################################################################################
class dTwistCtrl(Ctrl):
    def __init__(self):
        Ctrl.__init__(self)

    @abstractmethod
    def update(self, pos, vel, rstate, dt):
        pass


class KpCtrl(dTwistCtrl):
    def __init__(self, goal, Kp, Kd=None):
        dTwistCtrl.__init__(self)
        self._goal = goal
        self._Kp = Kp
        self._Kd = Kd if Kd is not None else 2*sqrt(Kp)

    def update(self, pos, vel, rstate, dt):
        if isinstance(self._goal, Joint):
            _goal = self._goal.gpos
        elif isinstance(self._goal, Frame):
            _goal = self._goal.pose
        else:
            _goal = self._goal

        self._error = diff(_goal, pos)
        return dot(self._Kp, diff(_goal, pos)) + dot(self._Kd, -vel)


class KpTrajCtrl(KpCtrl):
    def __init__(self, *args, **kargs):
        KpCtrl.__init__(self, *args, **kargs)
        self._counter = 0
        self._lim = -1
        self.set_goal(self._goal)

    def set_goal(self, new_goal):
        self._goal = new_goal
        if not all([v is None for v in self._goal]):
            self._lim = min([len(v) for v in self._goal if v is not None]) - 1
        else:
            self._lim = -1
        self._counter = 0

    def update(self, pos, vel, rstate, dt):
        idx = min(self._counter, self._lim)
        self._counter += 1

        pos_err = diff(self._goal[0][idx], pos) if \
                       self._goal[0] is not None else 0.
        vel_err = diff(self._goal[1][idx], vel) if \
                       self._goal[1] is not None else 0.
        acc     = self._goal[2][idx]            if \
                       self._goal[2] is not None else 0.

        return acc + dot(self._Kp, pos_err) + dot(self._Kd, vel_err)


class QuadraticCtrl(dTwistCtrl):
    def __init__(self, goal, QonR, horizon, dt):
        """ WARNING: THIS CONTROLLER WORKS ONLY IF dt IS CONSTANT!!!!!!
        """
        dTwistCtrl.__init__(self)
        self._goal = asarray(goal)
        self._QonR = QonR
        self._h    = int(horizon/dt)
        self._dt   = dt

    def set_goal(self, new_goal):
        self._goal = asarray(new_goal)



class ZMPCtrl(QuadraticCtrl):
    def __init__(self, zmp_traj, QonR, horizon, dt, cdof):
        QuadraticCtrl.__init__(self, zmp_traj, QonR, horizon, dt)

        self._Px = zeros((self._h, 3))
        self._Pu = zeros((self._h, self._h))
        self._Px[:, 0] = 1
        self._Px[:, 1] = arange(1, self._h+1)*self._dt
        self._range_N_dt_2_on_2 = (arange(1, self._h+1)*self._dt)**2/2.
        self._temp_Pu = zeros((self._h, self._h))
        for i in range(self._h):
            diag_i = (1 + 3*i + 3*i**2)*dt**3/6
            self._temp_Pu[range(i, self._h), range(self._h-i)] = diag_i

        self._cdof = cdof
        if 0 not in self._cdof:
            self._up = 0
        elif 1 not in self._cdof:
            self._up = 1
        elif 2 not in self._cdof:
            self._up = 2

        self._gravity   = 0.
        self._prev_vel = zeros(3)
        self._counter = 0

    def init(self, world, LQP_ctrl):
        for c in world.getcontrollers():
            if isinstance(c, WeightController):
                self._gravity = abs(c.gravity)

    def _get_com_hat_and_hong(self, pos, vel):
        dvel = (vel - self._prev_vel)/self._dt
        self._prev_vel = vel.copy()
        com_hat = array([pos, vel, dvel])
        com_hat = com_hat[:, self._cdof]

        hong = pos[self._up]/self._gravity

        return com_hat, hong

    def _fit_goal_for_horizon(self):
        goal = self._goal[self._counter:self._counter+self._h]
        self._counter += 1
        if len(goal) < self._h:
            final_value = self._goal[-1].reshape((1, 2))
            len_gap = self._h - len(goal)
            added_part = dot(ones((len_gap, 1)), final_value)
            goal = vstack([goal, added_part])
        return goal

    def _update_Px_and_Pu(self, hong):
        #self._Px[:,0] = 1                        #already computed in __init__
        #self._Px[:,1] = arange(1, self._h+1)*self._dt      #idem
        self._Px[:, 2] = self._range_N_dt_2_on_2 - hong
        self._Pu[:] = self._temp_Pu
        for i in range(self._h):
            self._Pu[range(i, self._h), range(self._h-i)] -= self._dt*hong

    def update(self, pos, vel, rstate, dt):
        assert(abs(dt-self._dt) < 1e-8)

        com_hat, hong = self._get_com_hat_and_hong(pos, vel)
        zmp_ref = self._fit_goal_for_horizon()
        self._update_Px_and_Pu(hong)

        ddV_com_cdof = get_quadratic_cmd(self._Px, self._Pu, \
                                         self._QonR, self._h, \
                                         com_hat, zmp_ref)
        
        dVcom_des = zeros(3)
        dVcom_des[self._cdof] = com_hat[2, :] + ddV_com_cdof[0] * dt
        return dVcom_des


################################################################################
#
# Wrench Controllers
#
################################################################################
class WrenchCtrl(Ctrl):
    def __init__(self):
        Ctrl.__init__(self)


class ValueCtrl(WrenchCtrl):
    def __init__(self, value):
        WrenchCtrl.__init__(self)

        self._value = asarray(value).flatten()

    def update(self, rstate, dt):
        return self._value


class TrajCtrl(WrenchCtrl):
    def __init__(self):
        WrenchCtrl.__init__(self)



