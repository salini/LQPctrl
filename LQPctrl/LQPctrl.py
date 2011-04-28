#coding=utf-8
#author=Joseph Salini
#date=21 april 2011


""" This module contains a controller which use a LQP to control a robot.

"""

from arboris.core import Controller
from numpy import zeros, hstack, vstack, dot


def lqpc_data(data={}):
    """Return a dictionnary of options for the LQP controller.

    :param options: the user-defined data
    :type options: dict

    :return: the default + user-defined data
    :rtype: dict

    """
    from numpy import eye
    _data = {'bodies'          : None,
             'Hfloor'          : eye(4),
             'cop constraints' : [],
               }
    _data.update(data)
    return _data

def lqpc_options(options={}):
    """Return a dictionnary of options for the LQP controller.

    :param options: the user-defined options
    :type options: dict

    :return: the default + user-defined options
    :rtype: dict

    """
    _options = {'pos horizon'  : None,
                'vel horizon'  : None,
                'npan'         : 8,
                'cost'         : 'normal',
                'base weights' : (1e-8, 1e-8, 1e-8),#(1e-1, 1e-1, 1e-1),#
                'solver'       : 'cvxopt',
                'formalism'    : 'dgvel chi', # | 'chi'
               }
    _options.update(options)
    return _options

def lqpc_solver_options(options={}):
    """Return a dictionnary of options for the used solver.

    :param options: the user-defined options
    :type options: dict

    :return: the default + user-defined options
    :rtype: dict

    """
    _options = {'show_progress' : True,
                'abstol'        : 1e-6,
                'reltol'        : 1e-6,
                'feastol'       : 1e-6,
                'maxiters'      : 100,
               }
    _options.update(options)
    return _options


class LQPcontroller(Controller):
    """

    """
    def __init__(self, gforcemax, qlim={}, vlim={}, tasks=[], events=[], data={}, options={}, solver_options={}, name=None):
        """ Initialize the instance of the LQP controller.

        :param gforcemax: max gforce (N or N/m) {'joint_name': max_gforce}
        :type gforcemax: dict

        :param qlim: position limits (rad) {'joint_name': (qmin,qmax)}
        :type qlim: dict

        :param vlim: velocity limits (rad/s) {'joint_name': (vmin,vmax)}
        :type vlim: dict

        :param events: a list of events (from LQPctrl.events)
        :type events: list

        :param tasks: a list of tasks (from LQPctrl.tasks)
        :type tasks: list

        :param data: a dictionnary of data interesting to all component of LQP
        :type data: dict

        :param options: a dictionnary of options for the LQPctrl
        :type options: dict

        :param solver_options: a dictionnary of options for the solver
        :type options: dict

        :param name: the name of the LQP controller

        """
        from arboris.core import NamedObjectsList
        #TODO: Make the assertion!!
        Controller.__init__(self, name=name)
        self.gforcemax = gforcemax
        self.qlim = dict(qlim)
        self.vlim = dict(vlim)
        self.events = NamedObjectsList(events)
        self.tasks = NamedObjectsList(tasks)
        self.data = lqpc_data(data)
        self.options = lqpc_options(options)
        self.solver_options = lqpc_solver_options(solver_options)


################################################################################
#
#   INIT FUNCTION AND SUB_FUNCTIONS
#
################################################################################

    def init(self, world):
        """ Initialize some internal properties before simulation.

        """
        self.world = world
        self.ndof = self.world.ndof

        self._init_bodies()
        self._init_WeightController()
        self._init_gforcemax()
        self._init_limits()
        self._init_constraints()
        self._init_solver()
        self._init_misc()

        for e in self.events:
            e.init(world, self)
        for t in self.tasks:
            t.init(world, self)


    def _init_bodies(self):
        """Register the bodies from options['bodies'].

        """
        self.bodies = self.data['bodies']
        if self.bodies is None:
            self.bodies = self.world.getbodies()
            self.bodies.remove(self.world.ground)


    def _init_WeightController(self):
        """Register WeightController if any.

        """
        from arboris.controllers import WeightController
        self.WeightController = None
        for c in self.world._controllers:
            if isinstance(c, WeightController):
                self.WeightController = c


    def _init_limits(self):
        """Initialize position, velocity and acceleration limits vectors.

        """
        from numpy import nan, isnan
        if len(self.qlim) == 0 and len(self.vlim) == 0:
            self._qlim = []
            self._vlim = []
        else:
            joints = self.world.getjoints()
            self._qlim = nan*zeros((self.ndof,2))
            for n, val in self.qlim.iteritems():
                self._qlim[joints[n].dof, 0] = val[0]
                self._qlim[joints[n].dof, 1] = val[1]
            self._vlim = nan*zeros((self.ndof,2))
            for n, val in self.vlim.iteritems():
                self._vlim[joints[n].dof, 0] = val[0]
                self._vlim[joints[n].dof, 1] = val[1]


    def _init_gforcemax(self):
        """Initialize gforcemax vector.

        """
        from numpy import array, zeros, nan, isnan
        j = self.world.getjoints()
        limits = nan*zeros(self.ndof)
        for n, val in self.gforcemax.iteritems():
            limits[j[n].dof] = val
        selected_dof = [i for i in range(self.ndof) if limits[i] if not isnan(limits[i])]
        self._gforcemax = limits[selected_dof]
        self.S = zeros((self.ndof, len(selected_dof)))          ### S the actuation matrix ###
        self.S[selected_dof, range(len(selected_dof))] = 1
        self.n_gforce = len(selected_dof)


    def _init_constraints(self):
        """ Register the available constraints.

        """
        from arboris.constraints import PointContact, BallAndSocketConstraint
        self.constraints = []
        self.is_enabled = {}
        for c in self.world._constraints:
            ### We only consider PointContact and BallAndSocketConstraint ###
            if isinstance(c, PointContact) or isinstance(c, BallAndSocketConstraint):
                self.constraints.append(c)
                self.is_enabled[c] = True

        self._mus = [getattr(c, '_mu', None) for c in self.constraints]
        self.n_fc = 3*len(self.constraints)

        if all([m is None for m in self._mus]):
            self._mus = []


    def _init_solver(self):
        """
        """
        from solver import init_solver
        init_solver(self.options['solver'], self.solver_options)

        if self.options['formalism'] == 'dgvel chi':
            self.n_problem = self.ndof + self.n_fc + self.n_gforce
        elif self.options['formalism'] == 'chi':
            self.n_problem = self.n_fc + self.n_gforce
        self.X_solution = zeros(self.n_problem)

        from numpy import ones, diag
        w_dgvel, w_fc, w_gforce = self.options['base weights']
        if self.options['cost'] is 'normal':
            if self.options['formalism'] == 'dgvel chi':
                Ediag = ones(self.ndof)*w_dgvel
            else:
                Ediag = zeros(0)
            Ediag = hstack((Ediag, ones(self.n_gforce)*w_gforce))
            Ediag = hstack((Ediag, ones(self.n_fc)*w_fc))
        self.E_base = diag(Ediag)
        self.f_base = zeros(self.n_problem)


    def _init_misc(self):
        """
        """
        self._performance = []


################################################################################
#
#   UPDATE FUNCTION
#
################################################################################
    def update(self, dt):
        """ Compute the minimal LQP cost and return (gforce, impedance)

        the LQP controller works as follow:

        * compute the state of the robot (CoM, ZMP, etc...)
        * update the events and tasks registered in the controller
        * extract the non-singular equality and inequality constraints
        * compute the cost function of the LQP from active tasks
        * return the computed gforce of the problem

        """
        from time import time as _time
        from solver import solve

        _performance = {}
        _tstart = _time()

        _t0 = _time() ### get robot state ###
        rstate = self._get_robot_state(dt)
        _performance['get robot state'] = _time() - _t0

        _t0 = _time() ### update tasks and events ###
        for e in self.events:
            e.update(rstate, dt)
        for t in self.tasks:
            t.update(rstate, dt)
        _performance['update tasks and events'] = _time() - _t0

        if self.world._current_time > 0 and len(self.tasks)>0:

            _t0 = _time() ### get LQP constraints from world configuration ###
            A, b, G, h = self._get_constraints(rstate, dt)
            _performance['get constraints'] = _time() - _t0

            _t0 = _time() ### sort the tasks by level ###
            sorted_tasks = self._get_sorted_tasks()
            _performance['sort tasks'] = _time() - _t0

            i=0
            _performance['get cost function'] = []
            _performance['solve'] = []
            _performance['constrain next level'] = []
            for tasks in sorted_tasks:

                _t0 = _time() ### compute cost function ###
                E_tasks, f_tasks = self._get_objective(tasks, dt)
                E = vstack((self.E_base, E_tasks))
                f = hstack((self.f_base, f_tasks))
                _performance['get cost function'].append(_time() - _t0)

                _t0 = _time() ### solve the LQP ###
                self.X_solution[:] = solve(E, f, G, h, A, b, self.options['solver'])
                _performance['solve'].append(_time() - _t0)

                _t0 = _time() ### concatenate solution with constraints to constrain next level ###
                if i<len(sorted_tasks)-1:
                    A = vstack((A, E_tasks))
                    b = hstack((b, dot(E_tasks, self.X_solution)))
                    i+=1
                _performance['constrain next level'].append(_time() - _t0)

            gforce = self._get_gforce_from_optimization(self.X_solution)
            _performance['total'] = _time() - _tstart
            self._performance.append(_performance)
        else:
            gforce = zeros(self.ndof)

        return (gforce, zeros((self.ndof, self.ndof)))


################################################################################
#
#   PRE RESOLUTION FUNCTIONS
#
################################################################################

    def _get_robot_state(self, dt):
        """ Compute the state of the robot.

        It returns a dictionary with the following string keys:

        * gpos: a dictionnary of the gpos {j: j.gpos}
        * gpos_reduce: an array with all the linear joint pos
        * TODO: rewrite correctly

        """
        from arboris.core import LinearConfigurationSpaceJoint
        from numpy import nan
        
        linear_gpos = nan*zeros(self.ndof)
        for j in self.world.getjoints():
            if isinstance(j, LinearConfigurationSpaceJoint):
                linear_gpos[j.dof] = j.gpos[:]

        gravity = zeros(self.ndof)
        if self.WeightController is not None:
            gravity[:] = self.WeightController.update()[0]

        Jc, dJc = self._get_Jc()

        rstate = {}
        rstate['X_solution'] = self.X_solution
        rstate['linear gpos'] = linear_gpos
        rstate['gvel'] = self.world.gvel
        rstate['Jc'] = Jc
        rstate['dJc'] = dJc
        rstate['gravity'] = gravity
        rstate['gravity_N'] = gravity-dot(self.world.nleffects, self.world.gvel)
        rstate['JcT_S'] = hstack((Jc.T, self.S))
        rstate['dJc_gvel'] = dot(dJc, self.world.gvel)

        if self.options['formalism'] == 'chi':
            from numpy.linalg import inv
            rstate['Minv'] = inv(self.world.mass)
            rstate['Minv(JcT_S)'] = dot(rstate['Minv'], rstate['JcT_S'])
            rstate['Minv(gravity_N)'] = dot(rstate['Minv'], rstate['gravity_N'])

        return rstate


    def _get_Jc(self):
        """Extract the Jacobian and dJacobian matrix of contact points.

        """
        from arboris.constraints import PointContact, BallAndSocketConstraint
        from arboris.homogeneousmatrix import adjoint, dAdjoint, iadjoint, inv
        Jc = zeros((self.n_fc, self.ndof))
        dJc = zeros((self.n_fc, self.ndof))
        i = 0
        for c in self.constraints:
            if c.is_enabled() and c.is_active() and self.is_enabled[c]:
                if isinstance(c, PointContact):
                    if c._frames[1].body in self.bodies:
                        Jc [3*i:(3*(i+1)), :] += c._frames[1].jacobian[3:6]
                        dJc[3*i:(3*(i+1)), :] += c._frames[1].djacobian[3:6]
                    if c._frames[0].body in self.bodies:
                        Jc [3*i:(3*(i+1)), :] -= c._frames[0].jacobian[3:6]
                        dJc[3*i:(3*(i+1)), :] -= c._frames[0].djacobian[3:6]
                elif isinstance(c, BallAndSocketConstraint):
                    H1_0   = dot(inv(c._frames[1].pose), c._frames[0].pose)
                    Ad1_0  = adjoint(H1_0)
                    Ad0_1  = iadjoint(H1_0)
                    T0_g_0 = c._frames[0].twist
                    T1_g_1 = c._frames[1].twist
                    T1_g_0 = dot(Ad0_1, T1_g_1)
                    T0_1_0 = T0_g_0 - T1_g_0
                    J0 = dot(Ad1_0, c._frames[0].jacobian)
                    J1 = c._frames[1].jacobian
                    dJ0 = dot(Ad1_0, c._frames[0].djacobian) + dot(dAdjoint(Ad1_0, T0_1_0), c._frames[0].jacobian)
                    dJ1 = c._frames[1].djacobian
                    Jc[3*i:(3*(i+1)), :]  = (J1[3:6] - J0[3:6])
                    dJc[3*i:(3*(i+1)), :] = (dJ1[3:6] - dJ0[3:6])
            i+=1
        return Jc, dJc



################################################################################
#
#   RESOLUTION LOOP FUNCTIONS
#
################################################################################

    def _get_constraints(self, rstate, dt):
        """Return the constraints of the problem.

        Each constraint is build as:

        * dot(A, [dgvel, gforce, fc]) = b
        * dot(G, [dgvel, gforce, fc]) <= h

        The equality constraints are the concatenation of:

        * equation of motion constr.
        * equation of contact point constr.

        The inequality constraints are the concatenation of:

        * upper torque limit constr.
        * lower torque limit constr.
        * friction cone limit constr.

        :return: the set of equalities and inequalities (A,b,G,h)

        """
        gpos = rstate['linear gpos']
        gvel = rstate['gvel']
        M = self.world.mass
        N = dot(self.world.nleffects, gvel)
        Jc = rstate['Jc']
        dJc = rstate['dJc']
        gravity = rstate['gravity']
        S = self.S
        hpos, hvel = max(self.options['pos horizon'], dt), max(self.options['vel horizon'], dt)
        const_activity = [c.is_enabled() and c.is_active() and self.is_enabled[c] for c in self.constraints]


        ####### Compute constraints #######
        from constraints import eq_motion, eq_contact_acc, ineq_gforcemax, ineq_joint_limits, ineq_friction
        
        equalities   = [(zeros((0, self.n_problem)), zeros(0))]
        inequalities = [(zeros((0, self.n_problem)), zeros(0))]
        
        inequalities.append( ineq_gforcemax(self._gforcemax, self.ndof, self.n_fc, self.options['formalism']) )
        if self.options['formalism'] == 'dgvel chi':
            equalities.append( eq_motion(M, rstate['JcT_S'], rstate['gravity_N']) )
        if len(self.constraints) > 0:
            equalities.append( eq_contact_acc(Jc, rstate['dJc_gvel'], self.n_problem, const_activity,
                                              self.options['formalism'], rstate['Minv(JcT_S)'], rstate['Minv(gravity_N)']) )
        if len(self._mus) > 0:
            inequalities.append( ineq_friction(self._mus, const_activity, self.options['npan'], self.ndof, self.n_problem, self.options['formalism']) )
        if len(self._qlim) > 0 or len(self._vlim) > 0:
            inequalities.append( ineq_joint_limits(self._qlim, self._vlim, gpos, gvel, hpos, hvel,
                                                   self.n_problem, self.options['formalism'], rstate['Minv(JcT_S)'], rstate['Minv(gravity_N)']) )

        eq_A  , eq_b   = zip(*equalities)
        ineq_G, ineq_h = zip(*inequalities)
        
        A = vstack(eq_A)
        b = hstack(eq_b)
        G = vstack(ineq_G)
        h = hstack(ineq_h)

        return A, b, G, h


    def _get_sorted_tasks(self):
        """Get of list of level-sorted tasks.

        :return: A list of tasks from low-level to high-level.
        :rtype: list(LQPctrl.tasks)

        """
        unsorted_tasks = list(self.tasks)
        sorted_tasks = []
        while unsorted_tasks:
            task = unsorted_tasks.pop(0)
            is_placed = False
            for i in range(len(sorted_tasks)):
                if task.level == sorted_tasks[i][0].level:
                    sorted_tasks[i].append(task)
                    is_placed = True
                    break
                if task.level < sorted_tasks[i][0].level:
                    sorted_tasks.insert(i, [task])
                    is_placed = True
                    break
            if not is_placed:
                sorted_tasks.append([task])
        return sorted_tasks


    def _get_objective(self, tasks, dt):
        """Compute a cost function, an objective from a set of tasks.

        """
        E = zeros((0, self.n_problem))
        f = zeros(0)

        for t in tasks:
            if t.is_active:
                E = vstack( (E, t.weight*t.E) )
                f = hstack( (f, t.weight*t.f) )
        return E, f



################################################################################
#
#   POST RESOLUTION FUNCTIONS
#
################################################################################
    def _get_gforce_from_optimization(self, X_solution):
        """ Return the interesting part (the torque) of the solution vector.
        
        """
        if self.options['formalism'] == 'dgvel chi':
            return dot(self.S, X_solution[(self.ndof+self.n_fc):])
        elif self.options['formalism'] == 'chi':
            return dot(self.S, X_solution[self.n_fc:])


    def get_performance(self):
        """
        """
        from numpy import sum, mean
        perf = {}
        for n in self._performance[0]:
            perf[n] = mean([sum(p[n]) for p in self._performance])
        return perf



