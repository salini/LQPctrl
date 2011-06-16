#coding=utf-8
#author=Joseph Salini
#date=21 april 2011


""" This module contains a controller which use a LQP to control a robot.

"""

from arboris.core import Controller
from numpy import zeros, hstack, vstack, dot
from time import time as _time


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
                'base weights' : (1e-7, 1e-7, 1e-7),#(1e-1, 1e-1, 1e-1),#
                'solver'       : 'cvxopt',
                'cost'         : 'normal', # | 'wrench consistent'
                'norm'         : 'normal', # | 'inv(lambda)'
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
                'abstol'        : 1e-8,
                'reltol'        : 1e-7,
                'feastol'       : 1e-7,
                'maxiters'      : 100,
               }
    _options.update(options)
    return _options


class LQPcontroller(Controller):
    """

    """
    def __init__(self, gforcemax, dgforcemax={}, qlim={}, vlim={}, tasks=[], events=[], data={}, options={}, solver_options={}, name=None):
        """ Initialize the instance of the LQP controller.

        :param gforcemax: max gforce (N or Nm) {'joint_name': max_gforce}
        :type gforcemax: dict

        :param dgforcemax: max dgforce (N/s or Nm/s) {'joint_name': max_dgforce}
        :type dgforcemax: dict

        :param qlim: position limits (rad) {'joint_name': (qmin,qmax)}
        :type qlim: dict

        :param vlim: velocity limits (rad/s) {'joint_name': vmax}
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
        self.gforcemax = dict(gforcemax)
        self.dgforcemax = dict(dgforcemax)
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
            self._vlim = nan*zeros(self.ndof)
            for n, val in self.vlim.iteritems():
                self._vlim[joints[n].dof] = val


    def _init_gforcemax(self):
        """Initialize gforcemax vector.

        """
        from numpy import array, zeros, nan, isnan
        j = self.world.getjoints()
        limits  = nan*zeros(self.ndof)
        for n, val in self.gforcemax.iteritems():
            limits[j[n].dof] = val
        selected_dof = [i for i in range(self.ndof) if not isnan(limits[i])]
        self._gforcemax  = limits[selected_dof]

        if self.dgforcemax:
            dlimits = nan*zeros(self.ndof)
            for n, val in self.dgforcemax.iteritems():
                dlimits[j[n].dof] = self.dgforcemax[n]
            self._dgforcemax = dlimits[selected_dof]
            self._prec_gforce = zeros(len(self._dgforcemax))
        else:
            self._dgforcemax = None
            self._prec_gforce = None

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

        self.Jc = zeros((self.n_fc, self.ndof))
        self.dJc = zeros((self.n_fc, self.ndof))


    def _init_solver(self):
        """
        """
        from solver import init_solver
        init_solver(self.options['solver'], self.solver_options)

        self.cost = self.options['cost']
        self.norm = self.options['norm']
        self.formalism = self.options['formalism']
        self.n_chi = self.n_fc + self.n_gforce
        if self.options['formalism'] == 'dgvel chi':
            self.n_problem = self.ndof + self.n_chi
        elif self.options['formalism'] == 'chi':
            self.n_problem = self.n_chi
        self.X_solution = zeros(self.n_problem)
        self._gforce = zeros(self.ndof)

        from numpy import ones, diag
        w_dgvel, w_fc, w_gforce = self.options['base weights']
        #if self.options['cost'] is 'normal': #TODO: vary along 'cost'
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
        from solver import solve

        self._rec_performance = {}
        _tstart = _time()

        rstate = self._update_robot_state(dt)
        self._update_tasks_and_events(rstate, dt)

        if self.world._current_time > 0 and len(self.tasks)>0:

            A, b, G, h = self._get_constraints(rstate, dt)
            sorted_tasks = self._get_sorted_tasks()

            i=0
            self._rec_performance['get cost function'] = []
            self._rec_performance['solve'] = []
            self._rec_performance['constrain next level'] = []
            for tasks in sorted_tasks:

                _t0 = _time() ### compute cost function ###
                E_tasks, f_tasks = self._get_objective(tasks, dt)
                E = vstack((self.E_base, E_tasks))
                f = hstack((self.f_base, f_tasks))
                self._rec_performance['get cost function'].append(_time() - _t0)

                _t0 = _time() ### solve the LQP ###
                self.X_solution[:] = solve(E, f, G, h, A, b, self.options['solver'])
                self._rec_performance['solve'].append(_time() - _t0)

                _t0 = _time() ### concatenate solution with constraints to constrain next level ###
                if i<len(sorted_tasks)-1:
                    A = vstack((A, E_tasks))
                    b = hstack((b, dot(E_tasks, self.X_solution)))
                    i+=1
                self._rec_performance['constrain next level'].append(_time() - _t0)

            self._update_gforce_from_optimization(self.X_solution)
            self._rec_performance['total'] = _time() - _tstart
            self._performance.append(dict(self._rec_performance))
        else:
            self._gforce[:] = 0.

        return (self._gforce, zeros((self.ndof, self.ndof)))


################################################################################
#
#   PRE RESOLUTION FUNCTIONS
#
################################################################################

    def _update_robot_state(self, dt):
        """ Compute the state of the robot.

        It returns a dictionary with the following string keys:

        * gpos: a dictionnary of the gpos {j: j.gpos}
        * gpos_reduce: an array with all the linear joint pos
        * TODO: rewrite correctly

        """
        from arboris.core import LinearConfigurationSpaceJoint
        from numpy import nan
        from numpy.linalg import inv
        _start_time = _time() ### get robot state ###

        linear_gpos = nan*zeros(self.ndof)
        for j in self.world.getjoints():
            if isinstance(j, LinearConfigurationSpaceJoint):
                linear_gpos[j.dof] = j.gpos[:]

        gravity = zeros(self.ndof)
        if self.WeightController is not None:
            gravity[:] = self.WeightController.update()[0]

        self._update_Jc()

        rstate = {}
        rstate['X_solution'] = self.X_solution
        rstate['linear gpos'] = linear_gpos
        rstate['gvel'] = self.world.gvel
        rstate['Jc'] = self.Jc
        rstate['dJc'] = self.dJc
        rstate['Jchi.T'] = hstack((self.Jc.T, self.S))
        rstate['M'] = self.world.mass
        rstate['g-n'] = gravity - dot(self.world.nleffects, self.world.gvel)
        rstate['dJc.gvel'] = dot(self.dJc, self.world.gvel)

        if self.cost in ['wrench consistent', 'dtwist consistent', 'a/m']: #TODO: maybe find better?!?
            rstate['Minv'] = inv(self.world.mass)

        if self.norm in ['inv(lambda)']:
            if 'Minv' not in rstate:
                rstate['Minv'] = inv(self.world.mass)

        if self.norm in ['inv(ellipsoid)']:
            if 'Minv' not in rstate:
                rstate['Minv'] = inv(self.world.mass)
            rstate['Minv*Minv'] = dot(rstate['Minv'], rstate['Minv'])

        if self.formalism == 'chi':
            if 'Minv' not in rstate:
                rstate['Minv'] = inv(self.world.mass)
            rstate['Minv(Jchi.T)'] = dot(rstate['Minv'], rstate['Jchi.T'])
            rstate['Minv(g-n)'] = dot(rstate['Minv'], rstate['g-n'])

        self._rec_performance['update robot state'] = _time() - _start_time
        return rstate


    def _update_tasks_and_events(self, rstate, dt):
        """ Update the tasks and events registered in the LQP.
        """
        _start_time = _time() ### update tasks and events ###
        for e in self.events:
            e.update(rstate, dt)
        for t in self.tasks:
            t.update(rstate, dt)
        self._rec_performance['update tasks and events'] = _time() - _start_time


    def _update_Jc(self):
        """ Extract the Jacobian and dJacobian matrix of contact points.

        """
        from arboris.constraints import PointContact, BallAndSocketConstraint
        from arboris.homogeneousmatrix import adjoint, dAdjoint, iadjoint, inv
        self.Jc[:]  = 0.
        self.dJc[:] = 0.
        i = 0
        for c in self.constraints:
            if c.is_enabled() and c.is_active() and self.is_enabled[c]:
                if isinstance(c, PointContact):
                    if c._frames[1].body in self.bodies:
                        self.Jc [3*i:(3*(i+1)), :] += c._frames[1].jacobian[3:6]
                        self.dJc[3*i:(3*(i+1)), :] += c._frames[1].djacobian[3:6]
                    if c._frames[0].body in self.bodies:
                        self.Jc [3*i:(3*(i+1)), :] -= c._frames[0].jacobian[3:6]
                        self.dJc[3*i:(3*(i+1)), :] -= c._frames[0].djacobian[3:6]
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
                    self.Jc[3*i:(3*(i+1)), :]  = (J1[3:6] - J0[3:6])
                    self.dJc[3*i:(3*(i+1)), :] = (dJ1[3:6] - dJ0[3:6])
            i+=1



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
        from constraints import eq_motion, eq_contact_acc, ineq_gforcemax, ineq_joint_limits, ineq_friction
        _start_time = _time() ### get LQP constraints from world configuration ###

        Minv_Jchi_T = rstate.get('Minv(Jchi.T)') # these functions return None
        Minv_G_N = rstate.get('Minv(g-n)') # if Minv_Jchi_T & Minv_G_N not in rstate (not computed)
        const_activity = [c.is_enabled() and c.is_active() and self.is_enabled[c] for c in self.constraints]

        ####### Compute constraints #######
        equalities   = [(zeros((0, self.n_problem)), zeros(0))]
        inequalities = [(zeros((0, self.n_problem)), zeros(0))]

        inequalities.append( ineq_gforcemax(self._gforcemax, self._dgforcemax, dt, self._prec_gforce, self.ndof, self.n_fc, self.formalism) )
        if self.formalism == 'dgvel chi':
            M = rstate['M']
            Jchi_T = rstate['Jchi.T']
            G_N = rstate['g-n']
            equalities.append( eq_motion(M, Jchi_T, G_N) )
        if len(self.constraints) > 0:
            Jc = rstate['Jc']
            dJc_gvel = rstate['dJc.gvel']
            equalities.append( eq_contact_acc(Jc, dJc_gvel, self.n_problem, const_activity, self.formalism, Minv_Jchi_T, Minv_G_N) )
        if len(self._mus) > 0:
            npan = self.options['npan']
            inequalities.append( ineq_friction(self._mus, const_activity, npan, self.ndof, self.n_problem, self.formalism) )
        if len(self._qlim) > 0 or len(self._vlim) > 0:
            gpos = rstate['linear gpos']
            gvel = rstate['gvel']
            hpos, hvel = max(self.options['pos horizon'], dt), max(self.options['vel horizon'], dt)
            inequalities.append( ineq_joint_limits(self._qlim, self._vlim, gpos, gvel, hpos, hvel, self.n_problem, self.formalism, Minv_Jchi_T, Minv_G_N) )

        eq_A  , eq_b   = zip(*equalities)
        ineq_G, ineq_h = zip(*inequalities)
        
        A, b = vstack(eq_A), hstack(eq_b)
        G, h = vstack(ineq_G), hstack(ineq_h)

        self._rec_performance['get constraints'] = _time() - _start_time
        return A, b, G, h


    def _get_sorted_tasks(self):
        """Get of list of level-sorted tasks.

        :return: A list of tasks from low-level to high-level.
        :rtype: list(LQPctrl.tasks)

        """
        _start_time = _time() ### sort the tasks by level ###

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

        self._rec_performance['sort tasks'] = _time() - _start_time
        return sorted_tasks


    def _get_objective(self, tasks, dt):
        """Compute a cost function, an objective from a set of tasks.

        """
        E = zeros((0, self.n_problem))
        f = zeros(0)
        E = vstack([E]+[t.weight*t.E for t in tasks if t.is_active])
        f = hstack([f]+[t.weight*t.f for t in tasks if t.is_active])
        return E, f



################################################################################
#
#   POST RESOLUTION FUNCTIONS
#
################################################################################
    def _update_gforce_from_optimization(self, X_solution):
        """ Return the interesting part (the torque) of the solution vector.
        
        """
        if self.options['formalism'] == 'dgvel chi':
            self._gforce[:] = dot(self.S, X_solution[(self.ndof+self.n_fc):])
            self._prec_gforce = X_solution[(self.ndof+self.n_fc):]
        elif self.options['formalism'] == 'chi':
            self._gforce[:] = dot(self.S, X_solution[self.n_fc:])
            self._prec_gforce = X_solution[(self.n_fc):]


    def get_gforce(self):
        return self._gforce.copy()


    def get_performance(self):
        """
        """
        from numpy import sum, mean
        perf = {}
        if len(self._performance):
            for n in self._performance[0]:
                perf[n] = mean([sum(p[n]) for p in self._performance])
        return perf



