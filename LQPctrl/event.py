#coding=utf-8
#author=Joseph Salini
#date=7 july 2011

""" Module with all condition and execution classes to emulate Events.
"""

from abc import ABCMeta, abstractmethod
from arboris.core import NamedObject
from misc import extract_contact_points, convex_hull, is_in_convex_hull, \
                 com_properties, interpolate_log

common_flags = {}


class Cond:
    """ A class that represents a condition

    Return true if the condition is valid
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        """
        self.world = None
        self.LQP_ctrl = None

    def init(self, world, LQP_ctrl):
        pass

    @abstractmethod
    def update(self, rstate, dt):
        pass



class Exe:
    """ A class that represents an execution

    Normally, this class execute the update method
    if the conditions linked with an Event instance
    are fulfilled
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.world = None
        self.LQP_ctrl = None

    def init(self, world, LQP_ctrl):
        pass

    @abstractmethod
    def update(self, rstate, dt, is_cond_fulfilled):
        pass



class Event(NamedObject):
    """ An class Event that links conditions with executions
    """
    def __init__(self, cond, exe, name = None):
        """ An initialization of the Event instance

        inputs:
        cond: a list of conditions, should be Cond instance
        exe: a list of execution, should be Exe instance
        """
        NamedObject.__init__(self, name)
        self.is_active = True
        self._cond_is_fulfilled = False
        if not isinstance(cond, list):
            cond = [cond]
        for c in cond:
            if not isinstance(c, Cond):
                raise TypeError( \
                'The elements of the cond list must be Cond instance')
        self.cond = cond
        if not isinstance(exe, list):
            exe = [exe]
        for e in exe:
            if not isinstance(e, Exe):
                raise TypeError( \
                'The elements of the exe list must be Exe instance')
        self.exe = exe

    def init(self, world, LQP_ctrl):
        """ An initialization of the Event, to prepare the simulation
        """
#        self.world = world
        for cond in self.cond:
            cond.init(world, LQP_ctrl)
        for exe in self.exe:
            exe.init(world, LQP_ctrl)

    def update(self, rstate, dt, _rec_performance):
        """ Check if the conditions are fulfilled, execute if valid
        """
        if self.is_active:
            self._cond_is_fulfilled = True
            for cond in self.cond:
                self._cond_is_fulfilled &= cond.update(rstate, dt)
        else:
            self._cond_is_fulfilled = False

        # Even if the Event is not active, or the condition are not
        # fulfilled, exe, may be execute (for example if it takes
        # 3 steps to finish its execution.
        for exe in self.exe:
            exe.update(rstate, dt, self._cond_is_fulfilled)

    def cond_is_fulfilled(self):
        return self._cond_is_fulfilled

################################################################################
#                                                                              #
# CONDITIONS                                                                   #
#                                                                              #
################################################################################

class AtTime(Cond):
    def __init__(self, t):
        Cond.__init__(self)
        self.t = t
        self._is_done = False

    def init(self, world, LQP_ctrl):
        self. world = world

    def update(self, rstate, dt):
        if self.world.current_time >= self.t and self._is_done is False:
            self._is_done = True
            return True
        else:
            return False


class IfFlag(Cond):
    def __init__(self, flag_key, status):
        Cond.__init__(self)
        self.flag_key = flag_key
        self.status   = status

    def update(self, rstate, dt):
        if self.flag_key in common_flags:
            if common_flags[self.flag_key] == self.status:
                return True
            else:
                return False
        else:
            return False


class InConvexHull(Cond):
    """
    """
    def __init__(self, frames, point, dof, margin=0., duration=0.):
        Cond.__init__(self)
        self.frames = frames
        self.point_name = point
        self.dof = dof
        self.margin = margin
        self.duration = duration

    def init(self, world, LQP_ctrl):
        """
        """
        self.world = world
        self.LQP_ctrl = LQP_ctrl

    def update(self, rstate, dt):
        """
        """
        CH = convex_hull(extract_contact_points(self.frames, self.dof))

        if self.point_name in rstate:
            point = rstate[self.point_name]
        elif self.point_name == 'CoM':
            point = com_properties(self.LQP_ctrl.bodies, compute_J=False)
            rstate['CoM'] = point
        if len(point) == 3:
            point = point[self.dof]

        cond_validity = is_in_convex_hull(CH, point, self.margin)
        return cond_validity



################################################################################
#                                                                              #
# EXECUTIONS                                                                   #
#                                                                              #
################################################################################
class Printer(Exe):
    """ A Exe child which print a string if conditions are fulfilled
    """
    def __init__(self, sentence):
        """ An initialization of the Printer instance

        inputs:
        sentence: the string to display
        """
        Exe.__init__(self)
        self.sentence = sentence

    def update(self, rstate, dt, is_cond_fulfilled):
        """ Display the sentence if cond are fulfilled
        """
        if is_cond_fulfilled:
            print self.sentence


class SetFlag(Exe):
    def __init__(self, key, value):
        Exe.__init__(self)
        self.key = key
        self.value   = value

    def update(self, rstate, dt, is_cond_fulfilled):
        if is_cond_fulfilled:
            common_flags[self.key] = self.value



class ChangeWeight(Exe):
    """
    """
    def __init__(self, task, ew, duration, sw=None):
        """
        """
        Exe.__init__(self)
        self.task = task
        self._end_weight = ew
        self._start_weight = sw
        self._weight_sequence = []
        self.duration = duration
        self.counter = None
        

    def update(self, rstate, dt, is_cond_fulfilled):
        """
        """
        if is_cond_fulfilled:
            self.counter = 0
            if self._start_weight is None:
                sw = self.task.weight if self.task.weight > 1e-16 else 1e-16
            else:
                sw = self._start_weight if self._start_weight > 1e-16 else 1e-16
            ew = self._end_weight if self._end_weight > 1e-16 else 1e-16
            self._weight_sequence = interpolate_log(sw, ew, self.duration, dt)

        if self.counter is not None:
            if len(self._weight_sequence)>self.counter:
                self.task.set_weight(self._weight_sequence[self.counter])
                self.counter += 1
            else:
                self.counter = None


class ChangeLevel(Exe):
    """
    """
    def __init__(self, tasks, lvl):
        """
        """
        Exe.__init__(self)
        if not isinstance(tasks, list):
            tasks = [tasks]
        self.tasks = tasks
        self._lvl = lvl

    def update(self, rstate, dt, is_cond_fulfilled):
        """
        """
        if is_cond_fulfilled:
            for t in self.tasks:
                t.set_level(self._lvl)




class ChangeGoal(Exe):
    """ Change the goal of a task
    """
    def __init__(self, task, new_goal):
        """ An initialization of the ChangeGoal instance

        inputs:
        task: the task which we want to change the controller
        new_goal: the new goal
        """
        Exe.__init__(self)
        self.task = task
        self.new_goal = new_goal

    def update(self, rstate, dt, is_cond_fulfilled):
        """ Change the goal of the task if the conditions are fulfilled
        """
        if is_cond_fulfilled:
            self.task.counter = 0
            self.task.ctrl.set_goal(self.new_goal)



class Activator(Exe):
    """ A Exe Child class that set the activity of an element
    """
    def __init__(self, element, activity=True):
        """ An initialization of the Activator instance

        inputs:
        element: List with anything with an "is_active" method
        activity: - True to set the element activity to True
                  - False to set the element activity to False
                  - Anything else to toggle the element activity
        """
        Exe.__init__(self)
        if not isinstance(element, list):
            element = [element]
        for e in element:
            if not hasattr(e, 'is_active'):
                raise AttributeError( \
                'There is no "is_active" attribute in this element')
        self.element = element
        self.activity = activity

    def update(self, rstate, dt, is_cond_fulfilled):
        """ Set the activity of the element
        """
        if is_cond_fulfilled:
            for e in self.element:
                if self.activity is True or self.activity is False:
                    e.set_activity(self.activity)
                else:
                    e.set_activity(not e.is_active)


class ConstActivator(Exe):
    """ Set the activity of a constraint in the Arboris Simulation
    """
    def __init__(self, const, activity=True, in_lqp=False):
        """ An initialization of the ArborisConstActivator instance

        inputs:
        const: a list of constraint, should be arboris.core.Constraint instance
        activity: - True to set the element activity to True
                  - False to set the element activity to False
                  - Anything else to toggle the element activity
        """
        Exe.__init__(self)
        if not isinstance(const, list):
            const = [const]
        self.const = const
        self.activity = activity
        self.in_lqp = in_lqp

    def init(self, world, LQP_ctrl):
        self.LQP_ctrl = LQP_ctrl

    def update(self, rstate, dt, is_cond_fulfilled):
        """ Set the activity of the constraint in the simulation
        if the conditions are fulfilled.
        """
        if is_cond_fulfilled:
            if self.in_lqp is False:
                if self.activity is True:
                    for c in self.const:
                        c.enable()
                elif self.activity is False:
                    for c in self.const:
                        c.disable()
            else:
                for c in self.const:
                    self.LQP_ctrl.is_enabled[c] = self.activity


class DelayFlag(Exe):
    def __init__(self, key, value, delay):
        Exe.__init__(self)
        self.key   = key
        self.value = value
        self.delay = delay
        self._start_time = None

    def init(self, world, LQP_ctrl):
        self.world = world

    def update(self, rstate, dt, is_cond_fulfilled):
        if is_cond_fulfilled:
            self._start_time = self.world.current_time

        if self._start_time is not None and \
           self.world.current_time >= self._start_time+self.delay:
            common_flags[self.key] = self.value
            self._start_time = None


################################################################################
#                                                                              #
# SHORTCUTS                                                                    #
#                                                                              #
################################################################################
SetF = SetFlag
IfF  = IfFlag
DelayF = DelayFlag
InCH = InConvexHull
Prtr  = Printer
ChW   = ChangeWeight
ChG   = ChangeGoal
ChL   = ChangeLevel
AtT = AtTime
Act = Activator
CAct = ConstActivator
