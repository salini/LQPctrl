#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=25 may 2011


import unittest

from numpy import pi, arange

from common import Test_simple_3R

from arboris.core import simulate
from arboris.shapes import Plane, Point
from arboris.constraints import SoftFingerContact

from LQPctrl.LQPctrl   import LQPcontroller
from LQPctrl.task      import TorqueTask
from LQPctrl.task_ctrl import ValueCtrl


class Test_constraints(Test_simple_3R):

    def setUp(self):
        Test_simple_3R.setUp(self)

        self.joints["Shoulder"].gpos[:] = pi/2.
        self.joints["Elbow"].gpos[:]    = pi/6.
        self.joints["Wrist"].gpos[:]    = pi/3.
        self.tasks = [TorqueTask(self.joints["Shoulder"], ValueCtrl(.1))]
        self.gforcemax = {"Shoulder":10,"Elbow":5,"Wrist":2}
        self.dgforcemax = {}
        self.qlim = {}
        self.vlim = {}
        self.options = {}

#        plane = Plane(self.w.ground, (0,1,0,-.4), name="plane")
#        sphere = Point(self.w.getframes()['EndEffector'], "point")
#        self.w.register(SoftFingerContact((plane, sphere), 1.5, name="const"))
#        self.w.init()
#        self.const = self.w.getconstraints()

    def simulate(self):
        self.lqpc = LQPcontroller(self.gforcemax, self.dgforcemax, qlim=self.qlim, vlim=self.vlim, tasks=self.tasks, options=self.options, solver_options={"show_progress":False})
        self.w.register(self.lqpc)
        simulate(self.w, arange(0, .04, .01), [])


    def test_dgvelchi(self):
        self.options.update({'formalism':'dgvel chi'})
        self.simulate()

    def test_chi(self):
        self.options.update({'formalism':'chi'})
        self.simulate()




class Test_qlim(Test_constraints):

    def setUp(self):
        Test_constraints.setUp(self)
        self.qlim    = {"Shoulder":(-3*pi/2,3*pi/2),"Elbow":(-pi/2,pi/2),"Wrist":(-pi/2,pi/2)}

class Test_vlim(Test_constraints):

    def setUp(self):
        Test_constraints.setUp(self)
        self.vlim    = {"Shoulder":10.,"Elbow":10.,"Wrist":10.}

class Test_qlim_vlim(Test_constraints):

    def setUp(self):
        Test_constraints.setUp(self)
        self.qlim    = {"Elbow":(-pi/2,pi/2),"Wrist":(-pi/2,pi/2)}
        self.vlim    = {"Shoulder":10.,"Elbow":10.}

class Test_dgforcemax1(Test_constraints):

    def setUp(self):
        Test_constraints.setUp(self)
        self.gforcemax  = {"Shoulder":10,"Elbow":5,"Wrist":2}
        self.dgforcemax = {"Shoulder":10,"Elbow":5,"Wrist":2}

class Test_dgforcemax2(Test_constraints):

    def setUp(self):
        Test_constraints.setUp(self)
        self.gforcemax  = {"Elbow":5,"Wrist":2}
        self.dgforcemax = {"Shoulder":10,"Elbow":5}

class Test_dgforcemax3(Test_constraints):

    def setUp(self):
        Test_constraints.setUp(self)
        self.gforcemax  = {"Elbow":5,"Wrist":2}
        self.dgforcemax = {"Elbow":5}

class Test_dgforcemax4(Test_constraints):

    def setUp(self):
        Test_constraints.setUp(self)
        self.gforcemax  = {"Elbow":5,"Wrist":2}
        self.dgforcemax = {"Shoulder":10}



def suite():
    tests_suite = []
    for t in [Test_qlim, Test_vlim, Test_qlim_vlim, Test_dgforcemax1, Test_dgforcemax2, Test_dgforcemax3, Test_dgforcemax4]:
        tests_suite.append(unittest.TestLoader().loadTestsFromTestCase(t))

    return unittest.TestSuite(tests_suite)



if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
