#coding=utf-8
#author=Joseph Salini
#date=21 april 2011


from abc import ABCMeta, abstractmethod, abstractproperty
from arboris.core import Joint, Frame
from arboris.homogeneousmatrix import ishomogeneousmatrix
from numpy import zeros, asarray, dot


def diff(val1, val2):
    if ishomogeneousmatrix(asarray(val1)):
        v1 = quatpos(val1)
    else:
        v1 = asarray(val1)
    if ishomogeneousmatrix(asarray(val2)):
        v2 = quatpos(val2)
    else:
        v2 = asarray(val2)
    return v1 - v2



class Ctrl:
    def __init__(self):
        self._error = 0.

    def init(self, world, LQP_ctrl):
        pass

    @abstractmethod
    def update(self, rstate, dt):
        pass

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


class KpCtrl(dTwistCtrl):
    def __init__(self, goal, Kp, Kd=None):
        from numpy import sqrt
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


class KpTrajCtrl(dTwistCtrl):
    def __init__(self):
        pass



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



