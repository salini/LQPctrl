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
        self._cost = LQP_ctrl.cost
        self._norm = LQP_ctrl.norm
        self._formalism = LQP_ctrl.formalism
        self._E = zeros((len(self._cdof), LQP_ctrl.n_problem))
        self._f = zeros(len(self._cdof))
        self._error = 0.

        # some parameters that can be used, depend on cost, Task type and formalism
        self._J  = zeros((len(self._cdof), LQP_ctrl.ndof))
        self._dJ = zeros((len(self._cdof), LQP_ctrl.ndof))
        self._inv_lambda = zeros((len(self._cdof), len(self._cdof)))
        self._lambda     = zeros((len(self._cdof), len(self._cdof)))


    @abstractmethod
    def update(self, rstate, dt):
        self._compute_error(rstate['X_solution'])
        #
        # here: should update objective and jacobian of the task
        #
        self._update_E_f(rstate, dt)

    def _update_E_f(self, rstate, dt):
        """
        """
        self._update_lambda(rstate, dt)
        self._update_E_f_cost(rstate, dt)
        self._update_E_f_formalism(rstate, dt)

    @abstractmethod
    def _update_E_f_cost(self, rstate, dt):
        """
        """
        pass

    @abstractmethod
    def _update_E_f_formalism(self, rstate, dt):
        """
        """
        pass

    def _update_lambda(self, rstate, dt):
        """
        """
        from numpy.linalg import inv
        self._inv_lambda[:] = dot(self._J, dot(rstate['Minv'], self._J.T))
        self._lambda[:]     = inv(self._inv_lambda)

    def _compute_error(self, X_solution):
        self._error = norm(dot(self._E, X_solution) + self._f)


    @property
    def level(self):
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
        self._dVdes = zeros(len(self._cdof))

        self._E_dgvel = zeros((len(self._cdof), LQP_ctrl.ndof))
        self._f_dgvel = zeros(len(self._cdof))

    def update(self, rstate, dt):
        self._compute_error(rstate['X_solution'])
        if self._cost == 'wrench consistent' or self._norm == 'inv(lambda)':
            self._update_lambda(rstate, dt)
        #
        # here: should update objective and jacobian of the task
        #
        # finish the function with self._update_E_f(rstate, dt)

    def _update_E_f_cost(self, rstate, dt):
        """
        """
        if self._cost == 'normal':
            self._E_dgvel[:] = self._J
            self._f_dgvel[:] = dot(self._dJ, rstate['gvel']) - self._dVdes
        elif self._cost == 'wrench consistent':
            self._E_dgvel[:] = dot(self._lambda, self._J)
            self._f_dgvel[:] = dot(self._lambda, dot(self._dJ, rstate['gvel']) - self._dVdes)
        
        if self._norm == 'normal':
            pass
        elif self._norm == 'inv(lambda)':
            from numpy.linalg import cholesky
            L_T = cholesky(self._inv_lambda).T
            self._E_dgvel[:] = dot(L_T, self._E_dgvel[:])
            self._f_dgvel[:] = dot(L_T, self._f_dgvel[:])

    def _update_E_f_formalism(self, rstate, dt):
        """
        """
        if self._formalism == 'dgvel chi':
            self._E[:,0:self._ndof] = self._E_dgvel
            self._f[:]              = self._f_dgvel
        elif self._formalism == 'chi':
            self._E[:] = dot(self._E_dgvel, rstate['Minv(Jchi.T)'])
            self._f[:] = self._f_dgvel + dot(self._E_dgvel, rstate['Minv(g-n)'])


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

    def update(self, rstate, dt):
        """
        """
        dTwistTask.update(self, rstate, dt)
        
        gpos = self._joint.gpos
        gvel = self._joint.gvel
        cmd = self._ctrl.update(gpos, gvel, rstate, dt)
        self._dVdes[:] = cmd[self._cdof_in_joint]

        self._update_E_f(rstate, dt)


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
        self._Wdes = zeros(len(self._cdof))

        self._E_chi = zeros((len(self._cdof), LQP_ctrl.n_chi))
        self._f_chi = zeros(len(self._cdof))

    def update(self, rstate, dt):
        self._compute_error(rstate['X_solution'])
        if self._norm == 'inv(lambda)':
            self._update_lambda(rstate, dt)
        #
        # here: should update objective and jacobian of the task
        #
        # finish the function with self._update_E_f(rstate, dt)

    def _update_E_f_cost(self, rstate, dt):
        """
        """
        if self._cost == 'normal':
            self._E_chi[:] = dot(self._J, rstate['Jchi.T'])
            self._f_chi[:] =  - self._Wdes
        elif self._cost == 'wrench consistent':
            self._E_chi[:] = dot(self._J, rstate['Jchi.T'])
            self._f_chi[:] =  - self._Wdes
        
        if self._norm == 'normal':
            pass
        elif self._norm == 'inv(lambda)':
            from numpy.linalg import cholesky
            L_T = cholesky(self._inv_lambda).T
            self._E_chi[:] = dot(L_T, self._E_chi[:])
            self._f_chi[:] = dot(L_T, self._f_chi[:])

    def _update_E_f_formalism(self, rstate, dt):
        """
        """
        if self._formalism == 'dgvel chi':
            self._E[:,self._ndof: ] = self._E_chi
            self._f[:]              = self._f_chi
        elif self._formalism == 'chi':
            self._E[:] = self._E_chi
            self._f[:] = self._f_chi
    
    def update(self, rstate, dt):
        Task.update(self, rstate, dt)


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
        self._J = zeros((len(self._cdof), self._ndof))
        self._J[arange(len(self._cdof)), self._cdof] = 1

    def update(self, rstate, dt):
        """
        """
        self._compute_error(rstate['X_solution'])

        cmd = self._ctrl.update(rstate, dt)
        self._Wdes[:] = cmd[self._cdof_in_joint]

        self._update_E_f(rstate, dt)



class MultiTorqueTask(WrenchTask):
    def __init__(self):
        pass

class ForceTask(WrenchTask):
    def __init__(self):
        pass







