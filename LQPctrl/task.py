#coding=utf-8
#author=Joseph Salini
#date=21 april 2011

from abc import ABCMeta, abstractmethod, abstractproperty
from arboris.core import NamedObject, Joint, LinearConfigurationSpaceJoint, Frame, Body, Constraint
from arboris.constraints import BallAndSocketConstraint, PointContact, SoftFingerContact
from numpy import dot, zeros, arange, array, append
from numpy.linalg import norm


from task_ctrl import dTwistCtrl, WrenchCtrl

################################################################################
#
# Generic Tasks
#
################################################################################
class Task(NamedObject):
    """ A Task class which can control a part of the robot

    A task links a part of the robot, (i.e. a joint, a frame,
    the center of mass, etc.) with an objective defined by
    a controller.
    
    For the inherited class of Task you must write as following:
    
    class InheritedTask(Task):
        def __init__(self, ...):
            Task.__init(self, ...)
            [... your code ...]

        def init(self, world, LQP_ctrl):
            Task.init(self, world, LQP_ctrl)
            [... your code ...]

    and you must overwrite the methods:
    _update_matrices(self, rstate, dt)
    _update_lambda(self, rstate, dt)
    _update_E_f_cost(self, rstate, dt)
    _update_E_f_norm(self, rstate, dt)
    _update_E_f_formalism(self, rstate, dt)
    """
    __metaclass__ = ABCMeta


    def __init__(self, cdof=[], weight=1, level=0, is_active=True, name=None):
        """ Save the generic values of a task.

        :param cdof: the controlled dofs of the part.
        :type cdof: list

        :param weight: the weight (importance) of the task.
        :type weight: float in [0,1]

        :param level: the level (importance) of the task.
        :type level: integer

        :param is_active: set the activity of the task.
        :type is_active: True or False

        :param name: the name of the task.
        :type name: string
        """
        NamedObject.__init__(self, name)
        self._level = level
        self._weight = weight
        self._is_active = is_active
        self._cdof = cdof


    def init(self, world, LQP_ctrl):
        """ Init parameters and matrices for fast computation of the task.

        It sets: the cost, norm and formalism of the problem.

        It initializes: E, f & error, which define the tasks.
        """
        self._cost = LQP_ctrl.cost
        self._norm = LQP_ctrl.norm
        self._formalism = LQP_ctrl.formalism

        self._E = zeros((len(self._cdof), LQP_ctrl.n_problem))
        self._f = zeros(len(self._cdof))
        self._error = 0.

        self._inv_lambda    = zeros((len(self._cdof), len(self._cdof)))
        self._lambda        = zeros((len(self._cdof), len(self._cdof)))
        self._inv_ellipsoid = zeros((len(self._cdof), len(self._cdof)))
        self._ellipsoid     = zeros((len(self._cdof), len(self._cdof)))
        self._L_T           = zeros((len(self._cdof), len(self._cdof)))

        self._inv_lambda_is_already_computed    = False
        self._lambda_is_already_computed        = False
        self._inv_ellipsoid_is_already_computed = False
        self._ellipsoid_is_already_computed     = False


    def update(self, rstate, dt):
        """ Update the task parameters.

        It computes: 
        the task error,
        updates matrices,
        updates E & f before LQPcontroller compute the whole cost function
        of the problem.
        """
        self._update_error(rstate['X_solution'])
        self._update_matrices(rstate, dt)
        self._update_E_f(rstate, dt)


    def _update_error(self, X_solution):
        """ Compute the error of the previous time step.

        It compute norm(E.\chi* + f) where \chi* is the optimal solution of the
        previous problem.
        """
        self._error = norm(dot(self._E, X_solution) + self._f)


    @abstractmethod
    def _update_matrices(self, rstate, dt):
        """ Update the matrices (must be overwrite by inherited).

        Here, interesting matrices are updated, as jacobian, its derivative,
        the objective, etc...
        """
        pass


    def _update_E_f(self, rstate, dt):
        """ Update E & f which define the task.

        It computes:
        lambda, depend of the cost/norm/formalism,
        first  : E & f, depend on cost type,
        second : E & f, depend on norm,
        finally: E & f, depend on formalism.
        """
        self._raz_flags()
        self._update_E_f_cost(rstate, dt)
        self._update_E_f_norm(rstate, dt)
        self._update_E_f_formalism(rstate, dt)


    def _raz_flags(self):
        """ Update lambda, the task mass matrix in operational space.
        """
        self._inv_lambda_is_already_computed    = False
        self._lambda_is_already_computed        = False
        self._inv_ellipsoid_is_already_computed = False
        self._ellipsoid_is_already_computed     = False


    def _update_inv_lambda(self, rstate, dt):
        """
        """
        if not self._inv_lambda_is_already_computed:
            self._inv_lambda_is_already_computed = True
            self._inv_lambda[:] = dot(self._J, dot(rstate['Minv'], self._J.T))
            self._inv_lambda[:] = self._inv_lambda / norm(self._inv_lambda)


    def _update_lambda(self, rstate, dt):
        """
        """
        from numpy.linalg import inv, pinv, LinAlgError

        if not self._lambda_is_already_computed:
            self._lambda_is_already_computed = True
            self._update_inv_lambda(rstate, dt)
            try:
                self._lambda[:] = inv(self._inv_lambda)
            except LinAlgError:
                self._lambda[:] = pinv(self._inv_lambda)


    def _update_inv_ellipsoid(self, rstate, dt):
        """
        """
        if not self._inv_ellipsoid_is_already_computed:
            self._inv_ellipsoid_is_already_computed = True
            self._inv_ellipsoid[:] = dot(self._J, dot(rstate['Minv*Minv'], self._J.T))
            self._inv_ellipsoid[:] = self._inv_ellipsoid / norm(self._inv_ellipsoid)


    def _update_ellipsoid(self, rstate, dt):
        """
        """
        from numpy.linalg import inv, pinv, LinAlgError

        if not self._ellipsoid_is_already_computed:
            self._ellipsoid_is_already_computed = True
            self._update_inv_ellipsoid()
            try:
                self._ellipsoid[:] = inv(self._inv_ellipsoid)
            except LinAlgError:
                self._ellipsoid[:] = pinv(self._inv_ellipsoid)


    def _update_L_T_for_normalization(self, norm_matrix):
        """
        """
        from numpy import diag, sqrt
        from numpy.linalg import cholesky, svd, LinAlgError
        try:
            self._L_T[:] = cholesky(norm_matrix).T
        except LinAlgError:
            u,s,vh = svd(norm_matrix, full_matrices=False)
            self._L_T[:] = dot(diag(sqrt(s)), u.T)


    @abstractmethod
    def _update_E_f_cost(self, rstate, dt):
        """ Update E & f, depend on cost (must be overwrite by inherited).
        """
        pass


    @abstractmethod
    def _update_E_f_norm(self, rstate, dt):
        """ Update E & f, depend on norm (must be overwrite by inherited).
        """
        pass


    @abstractmethod
    def _update_E_f_formalism(self, rstate, dt):
        """ Update E & f, depend on formalism (must be overwrite by inherited).
        """
        pass


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
    """ TODO.
    """


    def __init__(self, *args, **kwargs):
        """ TODO.
        """
        Task.__init__(self, *args, **kwargs)


    def init(self, world, LQP_ctrl):
        """ Initialize a dTwistTask.

        A generic dTwistTask is written as following:
                                     |dgvel|
        T = norm( [E_dgvel E_chi(=0)]| chi | + f )
        
        as it is a dTwistTask, we don't care about E_chi, and we just need
        informations about the jacobian of the task (J), its derivative (dJ)
        and the objective (dVdes).
        """
        Task.init(self, world, LQP_ctrl)

        self._ndof = LQP_ctrl.ndof
        self._J  = zeros((len(self._cdof), LQP_ctrl.ndof))
        self._dJ = zeros((len(self._cdof), LQP_ctrl.ndof))
        self._dVdes = zeros(len(self._cdof))

        self._E_dgvel = zeros((len(self._cdof), LQP_ctrl.ndof))
        self._f_dgvel = zeros(len(self._cdof))


    def _update_E_f_cost(self, rstate, dt):
        """ Update E & f cost for a dTwistTask.

        If the cost is 'normal':
        T = norm( J.dgvel + dJ.gvel - dVdes) (whatever the norm)
        So E_dgvel = J and f = dJ.gvel - dVdes.

        If the cost is 'wrench consistent':
        T = norm( \lambda.(J.dgvel + dJ.gvel - dVdes) ) (whatever the norm)
        So E_dgvel = \lambda.J and f = \lambda.(dJ.gvel - dVdes).
        """
        if self._cost == 'normal':
            self._E_dgvel[:] = self._J
            self._f_dgvel[:] = dot(self._dJ, rstate['gvel']) - self._dVdes

        elif self._cost == 'wrench consistent':
            self._update_lambda(rstate, dt)
            self._E_dgvel[:] = dot(self._lambda, self._J)
            self._f_dgvel[:] = dot(self._lambda, dot(self._dJ, rstate['gvel']) - self._dVdes)

        elif self._cost == 'dtwist consistent':
            self._E_dgvel[:] = self._J
            self._f_dgvel[:] = dot(self._dJ, rstate['gvel']) - self._dVdes

        elif self._cost == 'a/m':
            self._update_inv_lambda(rstate, dt)
            self._E_dgvel[:] = dot(self._inv_lambda, self._J)
            self._f_dgvel[:] = dot(self._inv_lambda, dot(self._dJ, rstate['gvel']) - self._dVdes)


    def _update_E_f_norm(self, rstate, dt):
        """ Update E & f norm for a dTwistTask.

        If the norm is 'normal':
        The cost of the task is T'.T = (E_dgvel.dgvel + f)'.(E_dgvel.dgvel + f)
        No change has to be done.

        If the nom is 'inv(lambda)':
        The norm of the task is T'.inv(lambda).T
        We use Cholesky: inv(lambda) = L.L' (inv(lambda) is symmetric)
        and the cost become: (E_dgvel.dgvel + f)'.L.L'(E_dgvel.dgvel + f)
        (L'.E_dgvel.dgvel + L'.f)'.(L'.E_dgvel.dgvel + L'.f)
        So E_dgvel = L'.E_dgvel and f = L'.f.
        """
        if self._norm == 'normal':
            return

        elif self._norm == 'inv(lambda)':
            self._update_inv_lambda(rstate, dt)
            self._update_L_T_for_normalization(self._inv_lambda)

        elif self._norm == 'inv(ellipsoid)':
            self._update_inv_ellipsoid(rstate, dt)
            self._update_L_T_for_normalization(self._inv_ellipsoid)

        self._E_dgvel[:] = dot(self._L_T, self._E_dgvel[:])
        self._f_dgvel[:] = dot(self._L_T, self._f_dgvel[:])


    def _update_E_f_formalism(self, rstate, dt):
        """ Update E & f formalism for a dTwistTask.

        If the formalism is 'dgvel chi':
        E = [E_dgvel 0] and f doesn't change.

        If the formalism is 'chi':
        We use the equation of motion: dgvel = Minv(Jchi'.\chi + g - n)
        So:
        E = E_dgvel.Minv.Jchi' and f = f + E_dgvel.Minv.(g - n)
        """
        if self._formalism == 'dgvel chi':
            self._E[:,0:self._ndof] = self._E_dgvel
            self._f[:]              = self._f_dgvel
        elif self._formalism == 'chi':
            self._E[:] = dot(self._E_dgvel, rstate['Minv(Jchi.T)'])
            self._f[:] = self._f_dgvel + dot(self._E_dgvel, rstate['Minv(g-n)'])


class JointTask(dTwistTask):
    """ TODO.
    """


    def __init__(self, joint, ctrl, *args, **kwargs):
        """ TODO.
        """
        dTwistTask.__init__(self, *args, **kwargs)

        self._joint = joint
        self._ctrl = ctrl
        assert(isinstance(joint, Joint))
        assert(isinstance(ctrl, dTwistCtrl))

        if self._cdof == []:
            self._cdof = arange(self._joint.ndof)
        else:
            assert(len(self._cdof) <= self._joint.ndof)
            assert(all([cd < self._joint.ndof for cd in self._cdof]))
        self._cdof_in_joint = list(self._cdof)


    def init(self, world, LQP_ctrl):
        """
        """
        dTwistTask.init(self, world, LQP_ctrl)
        self._ctrl.init(world, LQP_ctrl)

        joint_dofs_in_world = arange(self._ndof)[self._joint.dof]
        self._cdof = joint_dofs_in_world[self._cdof_in_joint]
        self._J[arange(len(self._cdof)), self._cdof] = 1


    def _update_matrices(self, rstate, dt):
        """
        """
        gpos = self._joint.gpos
        gvel = self._joint.gvel
        cmd = self._ctrl.update(gpos, gvel, rstate, dt)
        self._dVdes[:] = cmd[self._cdof_in_joint]


class FrameTask(dTwistTask):
    """ TODO.
    """


    def __init__(self, frame, ctrl, *args, **kwargs):
        """ TODO.
        """
        dTwistTask.__init__(self, *args, **kwargs)

        self._frame = frame
        self._ctrl = ctrl
        assert(isinstance(frame, Frame))
        assert(isinstance(ctrl, dTwistCtrl))

        if self._cdof == []:
            self._cdof = arange(6)
        else:
            assert(len(self._cdof) <= 6)
            assert(all([cd < 6 for cd in self._cdof]))


    def init(self, world, LQP_ctrl):
        """
        """
        dTwistTask.init(self, world, LQP_ctrl)
        self._ctrl.init(world, LQP_ctrl)


    def _update_matrices(self, rstate, dt):
        """
        """
        from arboris.homogeneousmatrix import adjoint, dAdjoint
        H_0_f = self._frame.pose
        HR_0_f = H_0_f.copy()
        HR_0_f[0:3, 3] = 0
        AdR_0_f = adjoint(HR_0_f)
        T_f_0_f = self._frame.twist #WARNING: twist relative to its frame
        T_f_0_0 = dot(AdR_0_f, T_f_0_f)
        T_f_0_f_only_rot = T_f_0_f.copy()
        T_f_0_f_only_rot[3:6] = 0.
        dAdR_0_f = dAdjoint(AdR_0_f, T_f_0_f_only_rot)

        J  = dot(AdR_0_f, self._frame.jacobian)
        dJ = dot(AdR_0_f, self._frame.djacobian) + dot(dAdR_0_f, self._frame.jacobian)

        gpos = self._frame.pose
        gvel = T_f_0_0
        cmd = self._ctrl.update(gpos, gvel, rstate, dt)

        self._J[:]     = J[self._cdof, :]
        self._dJ[:]    = dJ[self._cdof, :]
        self._dVdes[:] = cmd[self._cdof]



class MultiJointTask(dTwistTask):
    """ TODO.
    """


    def __init__(self, joints, ctrl, *args, **kwargs):
        """ TODO.
        """
        dTwistTask.__init__(self, *args, **kwargs)

        self._joints = joints
        self._ctrl = ctrl

        assert(hasattr(joints, "__iter__"))
        for j in joints:
            assert(isinstance(j, LinearConfigurationSpaceJoint))
        assert(isinstance(ctrl, dTwistCtrl))

        joints_ndof = sum([j.ndof for j in self._joints])
        if self._cdof == []:
            self._cdof = arange(joints_ndof)
        else:
            assert(len(self._cdof) <= joints_ndof)
            assert(all([cd < joints_ndof for cd in self._cdof]))
        self._cdof_in_joints = list(self._cdof)


    def init(self, world, LQP_ctrl):
        """
        """
        dTwistTask.init(self, world, LQP_ctrl)
        self._ctrl.init(world, LQP_ctrl)

        joints_dofs_in_world = array([],dtype=int)
        for j in self._joints:
            joints_dofs_in_world = append(joints_dofs_in_world, arange(self._ndof)[j.dof])
        self._cdof = array(joints_dofs_in_world)[self._cdof_in_joints]
        self._J[arange(len(self._cdof)), self._cdof] = 1


    def _update_matrices(self, rstate, dt):
        """
        """
        gpos = zeros(0)
        gvel = zeros(0)
        for j in self._joints:
            gpos = append(gpos, j.gpos)
            gvel = append(gvel, j.gvel)
        cmd = self._ctrl.update(gpos, gvel, rstate, dt)
        self._dVdes[:] = cmd[self._cdof_in_joints]



class CoMTask(dTwistTask):
    """ TODO.
    """


    def __init__(self, bodies, ctrl, *args, **kwargs):
        """ TODO.
        """
        dTwistTask.__init__(self, *args, **kwargs)

        self._bodies = bodies
        self._ctrl = ctrl
        for b in self._bodies:
            assert(isinstance(b, Body))
        assert(isinstance(ctrl, dTwistCtrl))

        if self._cdof == []:
            self._cdof = arange(3)
        else:
            assert(len(self._cdof) <= 3)
            assert(all([cd < 3 for cd in self._cdof]))


    def init(self, world, LQP_ctrl):
        """
        """
        dTwistTask.init(self, world, LQP_ctrl)
        self._ctrl.init(world, LQP_ctrl)


    def _update_matrices(self, rstate, dt):
        """
        """
        from misc import com_properties

        com_gpos, J, dJ = com_properties(self._bodies)
        com_gvel = dot(J, rstate['gvel'])
        cmd = self._ctrl.update(com_gpos, com_gvel, rstate, dt)

        self._J[:]     = J[self._cdof, :]
        self._dJ[:]    = dJ[self._cdof, :]
        self._dVdes[:] = cmd[self._cdof]



class LQPCoMTask(CoMTask):
    """ TODO.
    """


    def __init__(self, ctrl, *args, **kwargs):
        """
        """
        CoMTask.__init__(self, [], ctrl, *args, **kwargs)


    def init(self, world, LQP_ctrl):
        """
        """
        CoMTask.init(self, world, LQP_ctrl)
        self.lqp_bodies = LQP_ctrl.bodies


    def _update_matrices(self, rstate, dt):
        """
        """
        from misc import com_properties

        if "Pcom" not in rstate:
            com_gpos, J, dJ = com_properties(self.lqp_bodies)
            com_gvel = dot(J, rstate['gvel'])
            rstate['Pcom']  = com_gpos
            rstate['Vcom']  = com_gvel
            rstate['Jcom']  = J
            rstate['dJcom'] = dJ

        cmd = self._ctrl.update(rstate['Pcom'], rstate['Vcom'], rstate, dt)

        self._J[:]     = rstate['Jcom'][self._cdof, :]
        self._dJ[:]    = rstate['dJcom'][self._cdof, :]
        self._dVdes[:] = cmd[self._cdof]




################################################################################
#
# Wrench/Torque Tasks
#
################################################################################
class WrenchTask(Task):
    """
    """


    def __init__(self, *args, **kwargs):
        """
        """
        Task.__init__(self, *args, **kwargs)


    def init(self, world, LQP_ctrl):
        """ Initialize a WrenchTask.

        A generic WrenchTask is written as following:
                                     |dgvel|
        T = norm( [E_dgvel(=0) E_chi]| chi | + f )
        
        as it is a WrenchTask, we don't care about E_dgvel,
        we just need informations about the jacobian of the task (J) if needed
        by the cost/norm/formalism, and the objective (Wdes).
        """
        Task.init(self, world, LQP_ctrl)

        self._ndof = LQP_ctrl.ndof
        self._S = zeros((len(self._cdof), LQP_ctrl.n_chi))
        self._J = zeros((len(self._cdof), LQP_ctrl.ndof))
        self._Wdes = zeros(len(self._cdof))

        self._E_chi = zeros((len(self._cdof), LQP_ctrl.n_chi))
        self._f_chi = zeros(len(self._cdof))


    def _update_E_f_cost(self, rstate, dt):
        """ Update E & f cost for a WrenchTask.

        If the cost is 'normal':
        

        If the cost is 'wrench consistent':
        
        """
        if self._cost == 'normal':
            self._E_chi[:] = self._S
            self._f_chi[:] = - self._Wdes

        elif self._cost == 'wrench consistent':
            self._E_chi[:] = self._S
            self._f_chi[:] = - self._Wdes

        elif self._cost == 'dtwist consistent':
            self._update_inv_lambda(rstate, dt)
            self._E_chi[:] = dot(self._inv_lambda, self._S)
            self._f_chi[:] = dot(self._inv_lambda, - self._Wdes)

        elif self._cost == 'a/m':
            self._update_inv_lambda(rstate, dt)
            inv_lambda2 = dot(self._inv_lambda, self._inv_lambda)
            self._E_chi[:] = dot(inv_lambda2, self._S)
            self._f_chi[:] = dot(inv_lambda2, - self._Wdes)


    def _update_E_f_norm(self, rstate, dt):
        """
        """
        if self._norm == 'normal':
            return

        elif self._norm == 'inv(lambda)':
            self._update_inv_lambda(rstate, dt)
            self._update_L_T_for_normalization(self._inv_lambda)

        elif self._norm == 'inv(ellipsoid)':
            self._update_inv_ellipsoid(rstate, dt)
            self._update_L_T_for_normalization(self._inv_ellipsoid)
            
        self._E_chi[:] = dot(self._L_T, self._E_chi[:])
        self._f_chi[:] = dot(self._L_T, self._f_chi[:])


    def _update_E_f_formalism(self, rstate, dt):
        """
        """
        if self._formalism == 'dgvel chi':
            self._E[:, self._ndof: ] = self._E_chi
            self._f[:]               = self._f_chi
        elif self._formalism == 'chi':
            self._E[:] = self._E_chi
            self._f[:] = self._f_chi



class TorqueTask(WrenchTask):
    """
    """


    def __init__(self, joint, ctrl, *args, **kwargs):
        """
        """
        WrenchTask.__init__(self, *args, **kwargs)

        self._joint = joint
        self._ctrl = ctrl
        assert(isinstance(joint, Joint))
        assert(isinstance(ctrl, WrenchCtrl))

        if self._cdof == []:
            self._cdof = arange(self._joint.ndof)
        else:
            assert(len(self._cdof) <= self._joint.ndof)
            assert(all([cd < self._joint.ndof for cd in self._cdof]))
        self._cdof_in_joint = list(self._cdof)


    def init(self, world, LQP_ctrl):
        """
        """
        WrenchTask.init(self, world, LQP_ctrl)
        self._ctrl.init(world, LQP_ctrl)

        joint_dofs_in_world = arange(self._ndof)[self._joint.dof]
        self._cdof = joint_dofs_in_world[self._cdof_in_joint]
        self._J[arange(len(self._cdof)), self._cdof] = 1

        S_gforce = LQP_ctrl.S[self._cdof, : ]
        self._S[:, LQP_ctrl.n_fc: ] = S_gforce


    def _update_matrices(self, rstate, dt):
        """
        """
        cmd = self._ctrl.update(rstate, dt)
        self._Wdes[:] = cmd[self._cdof_in_joint]



class ForceTask(WrenchTask):
    """
    """


    def __init__(self, constraint, ctrl, *args, **kwargs):
        """
        """
        WrenchTask.__init__(self, *args, **kwargs)

        self._constraint = constraint
        self._ctrl = ctrl
        assert(isinstance(constraint, Constraint))
        assert(isinstance(ctrl, WrenchCtrl))

        if self._cdof == []:
            self._cdof = arange(3)
        else:
            assert(len(self._cdof) <= 3)
            assert(all([cd < 3 for cd in self._cdof]))


    def init(self, world, LQP_ctrl):
        """
        """
        WrenchTask.init(self, world, LQP_ctrl)
        self._ctrl.init(world, LQP_ctrl)

        S_wrench = zeros((len(self._cdof), LQP_ctrl.n_fc))
        for i in range(len(LQP_ctrl.constraints)):
            if self._constraint is LQP_ctrl.constraints[i]:
                S_wrench[range(len(self._cdof)), (self._cdof+3*i)] = 1

        self._S[:, :LQP_ctrl.n_fc] = S_wrench


    def _update_matrices(self, rstate, dt):
        """
        """
        if 1:#self._norm == 'inv(lambda)':
            J = self._constraint.jacobian
            if isinstance(self._constraint, BallAndSocketConstraint):
                self._J[:] = J[self._cdof,:]
            elif isinstance(self._constraint, SoftFingerContact):
                self._J[:] = J[(self._cdof+1),:]
            elif isinstance(self._constraint, PointContact):
                self._J[:] = J[self._cdof,:]

        cmd = self._ctrl.update(rstate, dt)
        self._Wdes[:] = cmd[self._cdof]



class MultiTorqueTask(WrenchTask):
    """ TODO
    """


    def __init__(self, joints, ctrl, *args, **kwargs):
        """
        """
        WrenchTask.__init__(self, *args, **kwargs)

        self._joints = joints
        self._ctrl = ctrl

        assert(hasattr(joints, "__iter__"))
        for j in joints:
            assert(isinstance(j, LinearConfigurationSpaceJoint))
        assert(isinstance(ctrl, WrenchCtrl))

        joints_ndof = sum([j.ndof for j in self._joints])
        if self._cdof == []:
            self._cdof = arange(joints_ndof)
        else:
            assert(len(self._cdof) <= joints_ndof)
            assert(all([cd < joints_ndof for cd in self._cdof]))
        self._cdof_in_joints = list(self._cdof)


    def init(self, world, LQP_ctrl):
        """
        """
        WrenchTask.init(self, world, LQP_ctrl)
        self._ctrl.init(world, LQP_ctrl)

        joints_dofs_in_world = array([],dtype=int)
        for j in self._joints:
            joints_dofs_in_world = append(joints_dofs_in_world, arange(self._ndof)[j.dof])
        self._cdof = array(joints_dofs_in_world)[self._cdof_in_joints]
        self._J[arange(len(self._cdof)), self._cdof] = 1

        S_gforce = LQP_ctrl.S[self._cdof, : ]
        self._S[:, LQP_ctrl.n_fc: ] = S_gforce


    def _update_matrices(self, rstate, dt):
        """
        """
        cmd = self._ctrl.update(rstate, dt)
        self._Wdes[:] = cmd[self._cdof_in_joints]




