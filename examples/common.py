#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

from arboris.core import World, Observer
from arboris.robots import simplearm
from arboris.controllers import WeightController
from arboris.shapes import Plane, Point
from arboris.constraints import SoftFingerContact

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
    w.register(plane)
    eeframe = w.getframes()['EndEffector']
    sphere = Point(eeframe, "point")
    w.register(sphere)
    
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
        percent = round(v/perf['total']*100.,2)
        print k, '(', percent, '%): ', round(v*1000,2), 'ms'


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
    def __init__(self, joint):
        _Recorder.__init__(self)
        self.joint = joint

    def update(self, dt):
        self._record.append(self.joint.gpos.copy())


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


