#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

from arboris.core import World, Observer, SubFrame
from arboris.robots import simplearm, icub
from arboris.controllers import WeightController
from arboris.shapes import Plane, Point, Sphere, Box
from arboris.constraints import SoftFingerContact
from arboris.homogeneousmatrix import transl

from numpy import array, pi, mean


def create_icub_and_init(chair=False, gravity=False):
    ## CREATE THE WORLD
    w = World()
    w._up[:] = [0,0,1]
    icub.add(w)
    w.register(Plane(w.ground, (0,0,1,0), "floor"))
    
    if chair is True:
        w.register(Sphere(w.getbodies()['l_hip_2'], .0325, name='l_gluteal'))
        w.register(Sphere(w.getbodies()['r_hip_2'], .0325, name='r_gluteal'))
        w.register(Box(SubFrame(w.ground, transl(.2, 0, 0.26 )), (.075, .1, .02), name='chair'))
        w.register(Box(SubFrame(w.ground, transl(.255, 0, 0.36 )), (.02, .1, .1), name='chair_back'))

    ## INIT
    joints = w.getjoints()
    joints['root'].gpos[0:3,3] = [0,0,.598]
    joints['l_shoulder_roll'].gpos[:] = pi/8
    joints['r_shoulder_roll'].gpos[:] = pi/8
    joints['l_elbow_pitch'].gpos[:] = pi/8
    joints['r_elbow_pitch'].gpos[:] = pi/8
    joints['l_knee'].gpos[:] = -0.1
    joints['r_knee'].gpos[:] = -0.1
    joints['l_ankle_pitch'].gpos[:] = -0.1
    joints['r_ankle_pitch'].gpos[:] = -0.1

    shapes = w.getshapes()
    floor_const = [SoftFingerContact((shapes[s], shapes['floor']), 1.5, name=s)for s in ['lf1', 'lf2', 'lf3', 'lf4', 'rf1', 'rf2', 'rf3', 'rf4']]
    for c in floor_const:
        w.register(c)

    if chair is True:
        chair_const = [SoftFingerContact((shapes[s], shapes['chair']), 1.5, name=s)for s in ['l_gluteal', 'r_gluteal']]
        for c in chair_const:
            w.register(c)

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


class RecordZMPPosition(_Recorder):
    def __init__(self, bodies):
        _Recorder.__init__(self)
        self.bodies = bodies

    def init(self, world, timeline):
        from arboris.controllers import WeightController
        self.w = world
        self._prev_gvel = world.gvel
        self.g = 0
        for c in world._controllers:
            if isinstance(c, WeightController):
                self.g = c.gravity * world._up

    def update(self, dt):
        from LQPctrl.misc import zmp_position
        dgvel = (self.w.gvel - self._prev_gvel)/dt
        self._record.append(zmp_position(self.bodies, self.g, self.w.gvel, dgvel, self.w._up))
        self._prev_gvel = self.w.gvel


