#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

import unittest

from numpy import arange

from arboris.core import World, simulate
from arboris.robots.simplearm import add_simplearm

from LQPctrl.LQPctrl   import LQPcontroller

class Test_3R_LQPCtrl(unittest.TestCase):

    def setUp(self):
        self.w = World()
        add_simplearm(self.w)
        self.joints = self.w.getjoints()
        self.frames = self.w.getframes()
        self.joints["Shoulder"].gpos[:] = .5
        self.joints["Elbow"].gpos[:]    = .5
        self.joints["Wrist"].gpos[:]    = .5
        self.w.update_dynamic()
        self.gforcemax = {"Shoulder":10,"Elbow":5,"Wrist":2}
        self.tasks = []


    def simulate(self, options):
        self.lqpc = LQPcontroller(self.gforcemax, tasks=self.tasks, options=options, solver_options={"show_progress":False})
        self.w.register(self.lqpc)
        simulate(self.w, arange(0, .04, .01), [])


    def test_normal_normal_dgvelchi(self):
        options = {'cost': 'normal', 'norm':'normal', 'formalism':'dgvel chi'}
        self.simulate(options)

    def test_wrench_normal_dgvelchi(self):
        options = {'cost': 'wrench consistent', 'norm':'normal', 'formalism':'dgvel chi'}
        self.simulate(options)

    def test_normal_invlambda_dgvelchi(self):
        options = {'cost': 'normal', 'norm':'inv(lambda)', 'formalism':'dgvel chi'}
        self.simulate(options)

    def test_wrench_invlambda_dgvelchi(self):
        options = {'cost': 'wrench consistent', 'norm':'inv(lambda)', 'formalism':'dgvel chi'}
        self.simulate(options)

    def test_normal_normal_chi(self):
        options = {'cost': 'normal', 'norm':'normal', 'formalism':'chi'}
        self.simulate(options)

    def test_wrench_normal_chi(self):
        options = {'cost': 'wrench consistent', 'norm':'normal', 'formalism':'chi'}
        self.simulate(options)

    def test_normal_invlambda_chi(self):
        options = {'cost': 'normal', 'norm':'inv(lambda)', 'formalism':'chi'}
        self.simulate(options)

    def test_wrench_invlambda_chi(self):
        options = {'cost': 'wrench consistent', 'norm':'inv(lambda)', 'formalism':'chi'}
        self.simulate(options)



from arboris.shapes import Plane, Point
from arboris.constraints import SoftFingerContact

class Test_3R_with_Plane_LQPCtrl(Test_3R_LQPCtrl):

    def setUp(self):
        Test_3R_LQPCtrl.setUp(self)

        plane = Plane(self.w.ground, name="plane")
        sphere = Point(self.w.getframes()['EndEffector'], "point")
        self.w.register(SoftFingerContact((plane, sphere), 1.5, name="const"))
        self.w.init()
        self.const = self.w.getconstraints()


