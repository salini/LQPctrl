#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

import unittest

from arboris.core import World
from arboris.robots.simplearm import add_simplearm


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
        pass


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


