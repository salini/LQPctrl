#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

import unittest

from numpy import arange, eye

from arboris.core import simulate

from LQPctrl.LQPctrl   import LQPcontroller
from LQPctrl.task      import JointTask, FrameTask, TorqueTask
from LQPctrl.task_ctrl import KpCtrl   , ValueCtrl

from common import Test_3R_LQPCtrl

class Test_CostNormFormalism(Test_3R_LQPCtrl):

    def simulate(self, options):
        self.lqpc = LQPcontroller(self.gforcemax, tasks=self.tasks, options=options, solver_options={"show_progress":False})
        self.w.register(self.lqpc)
        simulate(self.w, arange(0, .04, .01), [])






class Test_JointTask(Test_CostNormFormalism):

    def setUp(self):
        Test_CostNormFormalism.setUp(self)
        self.tasks.append(JointTask(self.joints["Shoulder"], KpCtrl(0.1, 10), [], 1., 0, True))


class Test_FrameTask(Test_CostNormFormalism):

    def setUp(self):
        Test_CostNormFormalism.setUp(self)
        self.tasks.append(FrameTask(self.frames["EndEffector"], KpCtrl(eye(4), 10), [],1.,0, True))


class Test_TorqueTask(Test_CostNormFormalism):

    def setUp(self):
        Test_CostNormFormalism.setUp(self)
        self.tasks.append(TorqueTask(self.joints["Shoulder"], ValueCtrl(-.1), [],1.,0, True))







def suite():
    tests_suite = []
    for t in [Test_JointTask, Test_FrameTask, Test_TorqueTask]:
        tests_suite.append(unittest.TestLoader().loadTestsFromTestCase(t))

    return unittest.TestSuite(tests_suite)



if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
