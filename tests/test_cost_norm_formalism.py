#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

import unittest
from numpy import arange

from arboris.core import World, simulate
from arboris.robots.simplearm import add_simplearm

from LQPctrl.LQPctrl   import LQPcontroller
from LQPctrl.task      import JointTask, TorqueTask
from LQPctrl.task_ctrl import KpCtrl   , ValueCtrl


class Test_LQPctrl(unittest.TestCase):

    def setUp(self):
        self.w = World()
        add_simplearm(self.w)
        self.joints = self.w.getjoints()
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





class Test_JointTask(Test_LQPctrl):

    def setUp(self):
        Test_LQPctrl.setUp(self)
        self.tasks.append(JointTask(self.joints["Shoulder"], KpCtrl(0.1, 10), [], 1., 0, True))


class Test_TorqueTask(Test_LQPctrl):

    def setUp(self):
        Test_LQPctrl.setUp(self)
        self.tasks.append(TorqueTask(self.joints["Shoulder"], ValueCtrl(-.1), [],1.,0, True))







def suite():
    tests_suite = []
    for t in [Test_JointTask, Test_TorqueTask]:
        tests_suite.append(unittest.TestLoader().loadTestsFromTestCase(t))

    return unittest.TestSuite(tests_suite)



if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
