#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

from arboris.core import World, Observer
from arboris.robots import simplearm
from arboris.controllers import WeightController
from arboris.shapes import Plane, Point
from arboris.constraints import SoftFingerContact

from numpy import array, mean

def create_3r_and_init(gpos=(0,0,0), gvel=(0,0,0), gravity=False):
    ## CREATE THE WORLD
    w = World()
    simplearm.add_simplearm(w)

    ## INIT
    joints = w.getjoints()
    joints["Shoulder"].gpos[:] = gpos[0]
    joints["Elbow"].gpos[:]    = gpos[1]
    joints["Wrist"].gpos[:]    = gpos[2]
    joints["Shoulder"].gvel[:] = gvel[0]
    joints["Elbow"].gvel[:]    = gvel[1]
    joints["Wrist"].gvel[:]    = gvel[2]

    w.update_dynamic()

    ## CTRL
    if gravity:
        w.register(WeightController())

    return w


def add_plane_and_point_on_arm(w, coeff):
    plane = Plane(w.ground, coeff, "plane")
    sphere = Point(w.getframes()['EndEffector'], "point")
    
    w.register(SoftFingerContact((plane, sphere), 1.5, name="const"))
    
    w.init()



def get_usual_observers(w, scene=True, perf=True, h5=False, daenim=True):
    from arboris.visu_collada import write_collada_scene
    from arboris.observers import PerfMonitor, Hdf5Logger, DaenimCom
    obs = []
    if scene:
        write_collada_scene(w, "scene.dae", flat=True)
    if perf:
        obs.append(PerfMonitor(True))
    if h5:
        obs.append(Hdf5Logger("sim.h5", group="/", mode="w", flat=True))
    if daenim:
        obs.append(DaenimCom("daenim", "scene.dae", options="-fps 15",  flat=True))
    return obs

def print_lqp_perf(lqpc):
    print('-------------------------------------------------')
    perf = lqpc.get_performance()
    for k,v in perf.items():
        percent = round(mean(v)/mean(perf['total'])*100.,2)
        print k, '(', percent, '%): ', round(mean(v)*1000,2), 'ms'


from abc import ABCMeta, abstractmethod, abstractproperty
class _Recorder(Observer):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._record = []

    def init(self, world, timeline):
        pass

    @abstractmethod
    def update(self, dt):
        pass

    def finish(self):
        pass

    def get_record(self):
        return self._record


class RecordJointPosition(_Recorder):
    def __init__(self, joints):
        _Recorder.__init__(self)
        if not hasattr(joints, "__iter__"):
            joints = [joints]
        self.joints = joints

    def update(self, dt):
        rec = array([j.gpos.copy() for j in self.joints]).flatten()
        self._record.append(rec)


class RecordJointVelocity(_Recorder):
    def __init__(self, joints):
        _Recorder.__init__(self)
        if not hasattr(joints, "__iter__"):
            joints = [joints]
        self.joints = joints

    def update(self, dt):
        rec = array([j.gvel.copy() for j in self.joints]).flatten()
        self._record.append(rec)


class RecordGforce(_Recorder):
    def __init__(self, lqpc):
        _Recorder.__init__(self)
        self.lqpc = lqpc

    def update(self, dt):
        self._record.append(self.lqpc.get_gforce())


class RecordWrench(_Recorder):
    def __init__(self, const):
        _Recorder.__init__(self)
        self.const = const

    def update(self, dt):
        self._record.append(self.const._force.copy())


class RecordFramePosition(_Recorder):
    def __init__(self, frame):
        _Recorder.__init__(self)
        self.frame = frame

    def update(self, dt):
        self._record.append(self.frame.pose[0:3,3].copy())


class RecordCoMPosition(_Recorder):
    def __init__(self, bodies):
        _Recorder.__init__(self)
        self.bodies = bodies

    def update(self, dt):
        from LQPctrl.misc import com_properties
        self._record.append(com_properties(self.bodies, compute_J=False))


