#coding=utf-8
#author=Joseph Salini
#date=21 april 2011


""" This module contains a controller which use a LQP to control a robot.

"""
from arboris.core import Controller, NamedObjectsList, \
                         LinearConfigurationSpaceJoint
from arboris.controllers import WeightController
from arboris.collisions import choose_solver
from arboris.constraints import PointContact, BallAndSocketConstraint
from arboris.homogeneousmatrix import adjoint, dAdjoint, iadjoint

from numpy import zeros, hstack, vstack, dot, ones, diag, eye, nan, isnan, \
                  array, zeros, arange, sum as np_sum
from numpy.linalg import inv

from constraints import ineq_collision_avoidance, eq_motion, eq_contact_acc, \
                        ineq_gforcemax, ineq_joint_limits, ineq_friction
from solver import init_solver, solve


def lqpc_data(data=None):
    """Return a dictionnary of options for the LQP controller.

    :param options: the user-defined data
    :type options: dict

    :return: the default + user-defined data
    :rtype: dict

    """
    data = data or {}
    _data = {'bodies'          : None,
             'Hfloor'          : eye(4),
             'cop constraints' : [],
               }
    _data.update(data)
    return _data

def lqpc_options(options=None):
    """Return a dictionnary of options for the LQP controller.

    :param options: the user-defined options
    :type options: dict

    :return: the default + user-defined options
    :rtype: dict

    """
    options = options or {}
    _options = {'pos horizon'      : None,
                'vel horizon'      : None,
                'avoidance horizon': None,
                'avoidance margin' : 0.,
                'npan'         : 8,
                'base weights' : (1e-7, 1e-7, 1e-7),
                'solver'       : 'cvxopt',
                'cost'         : 'normal', # | 'wrench consistent'
                'norm'         : 'normal', # | 'inv(lambda)'
                'formalism'    : 'dgvel chi', # | 'chi'
               }
    _options.update(options)
    return _options

def lqpc_solver_options(options=None):
    """Return a dictionnary of options for the used solver.

    :param options: the user-defined options
    :type options: dict

    :return: the default + user-defined options
    :rtype: dict

    """
    options = options or {}
    _options = {'show_progress' : False,
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
    def __init__(self, gforcemax, dgforcemax=None, qlim=None, vlim=None, \
                       tasks=None, events=None, \
                       data=None, options=None, solver_options=None, name=None):
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
        Controller.__init__(self, name=name)
        self.gforcemax = dict(gforcemax)
        self.dgforcemax = dgforcemax or {}
        self.qlim = qlim or {}
        self.vlim = vlim or {}
        self.tasks = NamedObjectsList(tasks)  if tasks is not None \
                     else NamedObjectsList([])
        self.events = NamedObjectsList(events) if events is not None \
                      else NamedObjectsList([])
        self.data = lqpc_data(data)
        self.options = lqpc_options(options)
        self.solver_options = lqpc_solver_options(solver_options)

        self.world = None
        self.bodies = []
        self.WeightController = None
        self._qlim = []
        self._vlim = []
        self.ndof = 0
        self.S = zeros((0, 0))
        self.constraints = []
        self.is_enabled = {}
        self._mus = []
        self.n_fc = 0
        self.Jc = zeros((0, 0))
        self.dJc = zeros((0, 0))
        self._gforcemax = []
        self._dgforcemax = []
        self._prec_gforce = None
        self.n_chi = 0
        self.n_gforce = 0
        self.n_problem = 0
        self.collision_shapes = []
        self.X_solution = zeros(0)
        self._gforce = zeros(0)
        self.E_base = zeros((0, 0))
        self.f_base = zeros(0)
        self.cost = ""
        self.norm = ""
        self.formalism = ""
        self._performance = []


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
        self._init_collision_shapes()
        self._init_constraints()
        self._init_solver()

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
        for c in self.world.getcontrollers():
            if isinstance(c, WeightController):
                self.WeightController = c


    def _init_limits(self):
        """Initialize position, velocity and acceleration limits vectors.

        """
        if len(self.qlim) == 0 and len(self.vlim) == 0:
            self._qlim = []
            self._vlim = []
        else:
            joints = self.world.getjoints()
            self._qlim = nan*zeros((self.ndof, 2))
            for n, val in self.qlim.iteritems():
                self._qlim[joints[n].dof, 0] = val[0]
                self._qlim[joints[n].dof, 1] = val[1]
            self._vlim = nan*zeros(self.ndof)
            for n, val in self.vlim.iteritems():
                self._vlim[joints[n].dof] = val


    def _init_collision_shapes(self):
        """Initialize collision shape list

        """
        if 'collision shapes' in self.data:
            for couple_of_shapes in self.data['collision shapes']:
                self.collision_shapes.append(choose_solver(*couple_of_shapes))


    def _init_gforcemax(self):
        """Initialize gforcemax vector.

        """
        j = self.world.getjoints()
        limits  = nan*zeros(self.ndof)
        for n, val in self.gforcemax.iteritems():
            limits[j[n].dof] = val
        selected_dof = [i for i in arange(self.ndof) if not isnan(limits[i])]
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

        self.S = zeros((self.ndof, len(selected_dof))) # S = actuation matrix #
        self.S[selected_dof, arange(len(selected_dof))] = 1
        self.n_gforce = len(selected_dof)


    def _init_constraints(self):
        """ Register the available constraints.

        """
        for c in self.world.getconstraints():
            ### We only consider PointContact and BallAndSocketConstraint ###
            if isinstance(c, PointContact) or \
               isinstance(c, BallAndSocketConstraint):
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

        w_dgvel, w_fc, w_gforce = self.options['base weights']
        if self.options['formalism'] == 'dgvel chi':
            Ediag = ones(self.ndof)*w_dgvel
        else:
            Ediag = zeros(0)
        Ediag = hstack((Ediag, ones(self.n_gforce)*w_gforce))
        Ediag = hstack((Ediag, ones(self.n_fc)*w_fc))
        self.E_base = diag(Ediag)
        self.f_base = zeros(self.n_problem)



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

        rstate = self._update_robot_state(dt)
        self._update_tasks_and_events(rstate, dt)

        if self.world.current_time > 0 and len(self.tasks)>0:

            A, b, G, h = self._get_constraints(rstate, dt)
            sorted_tasks = self._get_sorted_tasks()

            i = 0
            for tasks in sorted_tasks:

                ### compute cost function ###
                E_tasks, f_tasks = self._get_objective(tasks, dt)
                E = vstack((self.E_base, E_tasks))
                f = hstack((self.f_base, f_tasks))

                ### solve the LQP ###
                self.X_solution[:] = solve(E, f, G, h, A, b, \
                                           self.options['solver'])

                ### add solution to constraints to constrain next level ###
                if i < len(sorted_tasks)-1:
                    A = vstack((A, E_tasks))
                    b = hstack((b, dot(E_tasks, self.X_solution)))
                    i += 1

            self._update_gforce_from_optimization(self.X_solution)
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

        if self.cost in ['wrench consistent', 'dtwist consistent', 'a/m']:
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

        return rstate


    def _update_tasks_and_events(self, rstate, dt):
        """ Update the tasks and events registered in the LQP.
        """
        for e in self.events:
            e.update(rstate, dt)
        for t in self.tasks:
            t.update(rstate, dt)


    def _update_Jc(self):
        """ Extract the Jacobian and dJacobian matrix of contact points.

        """
        self.Jc[:]  = 0.
        self.dJc[:] = 0.
        i = 0
        for c in self.constraints:
            if c.is_enabled() and c.is_active() and self.is_enabled[c]:
                frame0, frame1 = c._frames[0], c._frames[1]
                if isinstance(c, PointContact):
                    self.dJc[3*i:(3*(i+1)), :] = frame1.djacobian[3:6] - \
                                                 frame0.djacobian[3:6]
                    if frame1.body in self.bodies:
                        self.Jc [3*i:(3*(i+1)), :] += frame1.jacobian[3:6]
                    if frame0.body in self.bodies:
                        self.Jc [3*i:(3*(i+1)), :] -= frame0.jacobian[3:6]
                elif isinstance(c, BallAndSocketConstraint):
                    H1_0   = dot(inv(frame1.pose), frame0.pose)
                    Ad1_0  = adjoint(H1_0)
                    Ad0_1  = iadjoint(H1_0)
                    T0_g_0 = frame0.twist
                    T1_g_1 = frame1.twist
                    T1_g_0 = dot(Ad0_1, T1_g_1)
                    T0_1_0 = T0_g_0 - T1_g_0
                    J0 = dot(Ad1_0, frame0.jacobian)
                    J1 = frame1.jacobian
                    dJ0 = dot(Ad1_0, frame0.djacobian) + \
                          dot(dAdjoint(Ad1_0, T0_1_0), frame0.jacobian)
                    dJ1 = frame1.djacobian
                    self.Jc[3*i:(3*(i+1)), :]  = (J1[3:6] - J0[3:6])
                    self.dJc[3*i:(3*(i+1)), :] = (dJ1[3:6] - dJ0[3:6])
            i += 1



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
        ### get LQP constraints from world configuration ###

        Minv_Jchi_T = rstate.get('Minv(Jchi.T)') # these functions return None
        Minv_G_N = rstate.get('Minv(g-n)')       # if Minv_Jchi_T & Minv_G_N not
                                                 # in rstate (not computed)
        const_activity = [c.is_enabled() and c.is_active() and \
                          self.is_enabled[c] for c in self.constraints]

        ####### Compute constraints #######
        equalities   = [(zeros((0, self.n_problem)), zeros(0))]
        inequalities = [(zeros((0, self.n_problem)), zeros(0))]

        if self.world.current_time > 0:
            inequalities.append( ineq_gforcemax(self._gforcemax, \
                                 self._dgforcemax, dt, self._prec_gforce, \
                                 self.ndof, self.n_fc, self.formalism) )
        else:
            inequalities.append( ineq_gforcemax(self._gforcemax, None, None, \
                                 self.ndof, self.n_fc, self.formalism) )
        if self.formalism == 'dgvel chi':
            M = rstate['M']
            Jchi_T = rstate['Jchi.T']
            G_N = rstate['g-n']
            equalities.append( eq_motion(M, Jchi_T, G_N) )
        if len(self.constraints) > 0:
            Jc = rstate['Jc']
            dJc_gvel = rstate['dJc.gvel']
            equalities.append( eq_contact_acc(Jc, dJc_gvel, self.n_problem, \
                                              const_activity, self.formalism, \
                                              Minv_Jchi_T, Minv_G_N) )
        if len(self._mus) > 0:
            npan = self.options['npan']
            inequalities.append( ineq_friction(self._mus, const_activity, \
                                               npan, self.ndof, \
                                               self.n_problem, self.formalism) )
        if len(self._qlim) > 0 or len(self._vlim) > 0:
            gpos = rstate['linear gpos']
            gvel = rstate['gvel']
            hpos = max(self.options['pos horizon'], dt)
            hvel = max(self.options['vel horizon'], dt)
            inequalities.append( ineq_joint_limits(self._qlim, self._vlim, \
                                 gpos, gvel, hpos, hvel, self.n_problem, \
                                 self.formalism, Minv_Jchi_T, Minv_G_N) )
        # This part is about obstacle avoidance:
        if len(self.collision_shapes) > 0: 
            sdist, svel, J, dJ = self._extract_collision_shapes_data()
            hpos = max(self.options['avoidance horizon'], 10*dt)
            gvel = rstate['gvel']
            inequalities.append( ineq_collision_avoidance(sdist, svel, J, dJ, \
                                 gvel, hpos, self.n_problem, self.formalism, \
                                 Minv_Jchi_T, Minv_G_N) )

        eq_A  , eq_b   = zip(*equalities)
        ineq_G, ineq_h = zip(*inequalities)
        
        A, b = vstack(eq_A), hstack(eq_b)
        G, h = vstack(ineq_G), hstack(ineq_h)

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
            for i in arange(len(sorted_tasks)):
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
        E = vstack([E]+[t.weight*t.E for t in tasks if t.is_active])
        f = hstack([f]+[t.weight*t.f for t in tasks if t.is_active])
        return E, f


    def _extract_collision_shapes_data(self):
        """
        """
        cs = self.collision_shapes
        sdist = zeros(len(cs))
        svel  = zeros(len(cs))
        J      = zeros((len(cs), self.ndof))
        dJ     = zeros((len(cs), self.ndof))

        for i in arange(len(self.collision_shapes)):
            sdist[i], Hgc0, Hgc1 = cs[i][1](cs[i][0])
            sdist[i] -= self.options['avoidance margin']
            f0, f1 = cs[i][0][0].frame, cs[i][0][1].frame

            Hf0c0, Hf1c1 = dot(inv(f0.pose), Hgc0), dot(inv(f1.pose), Hgc1)
            tf0_g_f0, tf1_g_f1 = f0.twist, f1.twist
            Adc0f0, Adc1f1 = adjoint(inv(Hf0c0)), adjoint(inv(Hf1c1))
            tc0_g_c0, tc1_g_c1 = dot(Adc0f0, tf0_g_f0), dot(Adc1f1, tf1_g_f1)
            Jc0, Jc1 = dot(Adc0f0, f0.jacobian), dot(Adc1f1, f1.jacobian)
            #as Tc_f_c = 0, no motion between the 2 frames because same body
            dJc0, dJc1 = dot(Adc0f0, f0.jacobian), dot(Adc1f1, f1.jacobian)
            
            svel[i] = tc1_g_c1[5] - tc0_g_c0[5]
            dJ[i, :] = dJc1[5] - dJc0[5]
            if f1.body in self.bodies:
                J[i, :]  += Jc1[5]
            if f0.body in self.bodies:
                J[i, :]  -= Jc0[5]
        return sdist, svel, J, dJ



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


