#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

import unittest

import test_cost_norm_formalism

suite = unittest.TestSuite()
suite.addTest(test_cost_norm_formalism.suite())

unittest.TextTestRunner(verbosity=2).run(suite)
