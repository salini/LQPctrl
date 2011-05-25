#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

import unittest

from numpy import eye


from LQPctrl.task      import JointTask, MultiJointTask, FrameTask, TorqueTask, MultiTorqueTask, ForceTask
from LQPctrl.task_ctrl import KpCtrl   , ValueCtrl

from common import Test_3R_LQPCtrl, Test_3R_with_Plane_LQPCtrl




class Test_JointTask(Test_3R_LQPCtrl):

    def setUp(self):
        Test_3R_LQPCtrl.setUp(self)
        self.tasks.append(JointTask(self.joints["Shoulder"], KpCtrl(0.1, 10), [], 1., 0, True))


class Test_MultiJointTask(Test_3R_LQPCtrl):

    def setUp(self):
        Test_3R_LQPCtrl.setUp(self)
        self.tasks.append(MultiJointTask(self.joints, KpCtrl([.1,.1,.1], 10), [], 1., 0, True))


class Test_FrameTask(Test_3R_LQPCtrl):

    def setUp(self):
        Test_3R_LQPCtrl.setUp(self)
        self.tasks.append(FrameTask(self.frames["EndEffector"], KpCtrl(eye(4), 10), [],1.,0, True))


class Test_TorqueTask(Test_3R_LQPCtrl):

    def setUp(self):
        Test_3R_LQPCtrl.setUp(self)
        self.tasks.append(TorqueTask(self.joints["Shoulder"], ValueCtrl(-.1), [],1.,0, True))


class Test_MultiTorqueTask(Test_3R_LQPCtrl):

    def setUp(self):
        Test_3R_LQPCtrl.setUp(self)
        self.tasks.append(MultiTorqueTask(self.joints, ValueCtrl([.03,.02,.01]), [],1.,0, True))


class Test_ForceTask(Test_3R_with_Plane_LQPCtrl):

    def setUp(self):
        Test_3R_with_Plane_LQPCtrl.setUp(self)
        self.tasks.append(ForceTask(self.const['const'], ValueCtrl([0,0,0]), [],1.,0, True))







def suite():
    tests_suite = []
    for t in [Test_JointTask, Test_MultiJointTask, Test_FrameTask, Test_TorqueTask, Test_MultiTorqueTask, Test_ForceTask]:
        tests_suite.append(unittest.TestLoader().loadTestsFromTestCase(t))

    return unittest.TestSuite(tests_suite)



if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
