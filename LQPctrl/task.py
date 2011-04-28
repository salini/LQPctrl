#coding=utf-8
#author=Joseph Salini
#date=21 april 2011

from abc import ABCMeta, abstractmethod, abstractproperty
from arboris.core import NamedObject
from numpy import dot, zeros, arange
from numpy.linalg import norm


from task_ctrl import dTwistCtrl, WrenchCtrl

class Task(NamedObject):
    """ A Task class which can control a part of the robot

    A task links a part of the robot, like a joint, a frame,
    the center of mass, etc. with an objective defined by
    a controller.
    """
    __metaclass__ = ABCMeta

    def __init__(self, cdof=[], weight=1, level=0, is_active=True, name=None):
        NamedObject.__init__(self, name)
        self._level = level
        self._weight = weight
        self._is_active = is_active
        self._cdof = cdof

    def init(self, world, LQP_ctrl):
        self._cost = LQP_ctrl.options['cost']
        self._formalism = LQP_ctrl.options['formalism']
        self._E = zeros((len(self._cdof), LQP_ctrl.n_problem))
        self._f = zeros(len(self._cdof))

        self._prev_E = self._E.copy()
        self._prev_f = self._f.copy()
        self._error = 0.

    def update(self, rstate, dt):
        self._compute_error(rstate['X_solution'])

    def _compute_error(self, X_solution):
        self._error = norm(dot(self._E, X_solution) + self._f)


    @property
    def weight(self):
        return self._level

    @property
    def weight(self):
        return self._weight

    @property
    def is_active(self):
        return self._is_active

    @property
    def E(self):
        return self._E

    @property
    def f(self):
        return self._f

    @property
    def error(self):
        return self._error


################################################################################
#
# dTwist Tasks
#
################################################################################
class dTwistTask(Task):
    """
    """
    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
    
    def init(self, world, LQP_ctrl):
        Task.init(self, world, LQP_ctrl)

        self._ndof = LQP_ctrl.ndof
        self._J     = zeros((len(self._cdof), self._ndof))
        self._dJ    = zeros((len(self._cdof), self._ndof))
        self._dVdes = zeros(len(self._cdof))

    def update(self, rstate, dt):
        Task.update(self, rstate, dt)

    def _update_E_f(self, rstate, dt):
        """
        """
        if self._formalism == 'dgvel chi':
            self._E[:,0:self._ndof] = self._J
            self._f[:] = dot(self._dJ, rstate['gvel']) - self._dVdes
        elif self._formalism == 'chi':
            self._E[:] = dot(self._J, rstate['Minv(JcT_S)'])
            self._f[:] = dot(self._dJ, rstate['gvel']) - self._dVdes + dot(self._J, rstate['Minv(gravity_N)'])


class JointTask(dTwistTask):
    """
    """
    def __init__(self, joint, ctrl, *args, **kwargs):
        dTwistTask.__init__(self, *args, **kwargs)

        self._joint = joint
        self._ctrl = ctrl

        assert(isinstance(ctrl, dTwistCtrl))
        if self._cdof == []:
            self._cdof = arange(self._joint.ndof)
        else:
            assert(len(self._cdof) <= self._joint.ndof)
            assert(all([cd < self._joint.ndof for cd in self._cdof]))
        self._cdof_in_joint = list(self._cdof)


    def init(self, world, LQP_ctrl):
        dTwistTask.init(self, world, LQP_ctrl)
        self._ctrl.init(world, LQP_ctrl)

        joint_dofs_in_world = arange(self._ndof)[self._joint.dof]
        self._cdof = joint_dofs_in_world[self._cdof_in_joint]
        self._J[arange(len(self._cdof)), self._cdof] = 1

        if self._formalism == 'dgvel chi':
            self._E[:,0:self._ndof] = self._J


    def update(self, rstate, dt):
        """
        """
        dTwistTask.update(self, rstate, dt)
        gpos = self._joint.gpos
        gvel = self._joint.gvel

        cmd = self._ctrl.update(gpos, gvel, rstate, dt)
        print "ctrl err: ", self._ctrl.error[self._cdof_in_joint]

        self._dVdes[:] = cmd[self._cdof_in_joint]
        self._update_E_f(rstate, dt)


    def _update_E_f(self, rstate, dt):
        """
        """
        if self._formalism == "dgvel chi":
            self._f[:] = - self._dVdes
        elif self._formalism == 'chi':
            self._E[:] = dot(self._J, rstate['Minv(JcT_S)'])
            self._f[:] = dot(self._dJ, rstate['gvel']) - self._dVdes + dot(self._J, rstate['Minv(gravity_N)'])


class MultiJointTask(dTwistTask):
    def __init__(self):
        pass

class FrameTask(dTwistTask):
    def __init__(self):
        pass

################################################################################
#
# Wrench/Torque Tasks
#
################################################################################
class WrenchTask(Task):
    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)

    def init(self, world, LQP_ctrl):
        Task.init(self, world, LQP_ctrl)

        self._ndof = LQP_ctrl.ndof
        self._n_gforce = LQP_ctrl.n_gforce
        self._JS    = zeros((len(self._cdof), self._n_gforce))
        self._Wdes = zeros(len(self._cdof))

    def update(self, rstate, dt):
        Task.update(self, rstate, dt)

    def _update_E_f(self, rstate, dt):
        """
        """
        if self._formalism == 'dgvel chi':
            self._f[:] = - self._Wdes
        elif self._formalism == 'chi':
            self._f[:] =  - self._Wdes


class TorqueTask(WrenchTask):
    def __init__(self, joint, ctrl, *args, **kwargs):
        WrenchTask.__init__(self, *args, **kwargs)

        self._joint = joint
        self._ctrl = ctrl

        assert(isinstance(ctrl, WrenchCtrl))
        if self._cdof == []:
            self._cdof = arange(self._joint.ndof)
        else:
            assert(len(self._cdof) <= self._joint.ndof)
            assert(all([cd < self._joint.ndof for cd in self._cdof]))
        self._cdof_in_joint = list(self._cdof)

    def init(self, world, LQP_ctrl):
        WrenchTask.init(self, world, LQP_ctrl)
        self._ctrl.init(world, LQP_ctrl)

        joint_dofs_in_world = arange(self._ndof)[self._joint.dof]
        self._cdof = joint_dofs_in_world[self._cdof_in_joint]
        J = zeros((len(self._cdof), self._ndof))
        J[arange(len(self._cdof)), self._cdof] = 1
        self._JS = dot(J, LQP_ctrl.S)

        if self._formalism == 'dgvel chi':
            self._E[:,(LQP_ctrl.ndof+LQP_ctrl.n_fc):] = self._JS
        elif self._formalism == 'chi':
            self._E[:,LQP_ctrl.n_fc:] = self._JS

    def update(self, rstate, dt):
        """
        """
        WrenchTask.update(self, rstate, dt)
        cmd = self._ctrl.update(rstate, dt)
        print "up!! ", self._Wdes, cmd, self._cdof_in_joint
        self._Wdes[:] = cmd[self._cdof_in_joint]
        self._update_E_f(rstate, dt)



class MultiTorqueTask(WrenchTask):
    def __init__(self):
        pass

class ForceTask(WrenchTask):
    def __init__(self):
        pass







