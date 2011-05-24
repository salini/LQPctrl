#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

import unittest

from numpy import array, arange, eye
from numpy.linalg import norm

from arboris.core import simulate

from LQPctrl.LQPctrl   import LQPcontroller
from LQPctrl.task      import JointTask, FrameTask, TorqueTask
from LQPctrl.task_ctrl import KpCtrl   , ValueCtrl

from common import Test_3R_LQPCtrl

class Test_WithKpCtrl(Test_3R_LQPCtrl):

    def simulate(self, options):
        self.lqpc = LQPcontroller(self.gforcemax, tasks=self.tasks, options=options, solver_options={"show_progress":False})
        self.w.register(self.lqpc)
        simulate(self.w, arange(0, 4., .01), [])
        self.check_results()






class Test_JointKpCtrl(Test_WithKpCtrl):

    def setUp(self):
        Test_WithKpCtrl.setUp(self)
        self.tasks.append(JointTask(self.joints["Shoulder"], KpCtrl(0.1, 20), [], 1., 0, True))
        self.tasks.append(JointTask(self.joints["Elbow"], KpCtrl(0.1, 20), [], 1., 0, True))
        self.tasks.append(JointTask(self.joints["Wrist"], KpCtrl(0.1, 20), [], 1., 0, True))

    def check_results(self):
        val = norm(self.joints["Shoulder"].gpos - array([.1])) + \
              norm(self.joints["Elbow"].gpos - array([.1])) + \
              norm(self.joints["Wrist"].gpos - array([.1]))
        self.assertTrue(val <= 1e-3)


class Test_FrameKpCtrl(Test_WithKpCtrl):

    def setUp(self):
        Test_WithKpCtrl.setUp(self)
        self.tasks.append(FrameTask(self.frames["EndEffector"], KpCtrl(eye(4), 20), [3,4],1.,0, True))

    def check_results(self):
        self.assertTrue(norm(self.frames["EndEffector"].pose[0:3,3] - array([0.,0.,0.])) <= 1e-3)




def suite():
    tests_suite = []
    for t in [Test_JointKpCtrl, Test_FrameKpCtrl]:
        tests_suite.append(unittest.TestLoader().loadTestsFromTestCase(t))

    return unittest.TestSuite(tests_suite)



if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
