#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

from arboris.core import World, Observer
from arboris.robots import simplearm
from arboris.controllers import WeightController


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
        obs.append(DaenimCom("daenim", "scene.dae", options="-fps 15",  flat=True)) # -eye 1 0 0 -coi 0 0 0 -up 0 1 0
    return obs

def print_lqp_perf(lqpc):
    print('-------------------------------------------------')
    perf = lqpc.get_performance()
    for k,v in perf.items():
        percent = round(v/perf['total']*100.,2)
        print k, '(', percent, '%): ', round(v*1000,2), 'ms'

class RecordJointPosition(Observer):
    def __init__(self, joint):
        self.joint = joint
        self._record = []

    def init(self, world, timeline):
        pass

    def update(self, dt):
        self._record.append(self.joint.gpos.copy())

    def finish(self):
        pass

    def get_positions(self):
        return self._record


class RecordGforce(Observer):
    def __init__(self, lqpc):
        self.lqpc = lqpc
        self._record = []

    def init(self, world, timeline):
        pass

    def update(self, dt):
        self._record.append(self.lqpc.get_gforce())

    def finish(self):
        pass

    def get_gforce(self):
        return self._record
