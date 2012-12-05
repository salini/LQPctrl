#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

from common import create_3r_and_init, RecordJointPosition, RecordFramePosition

from arboris.core import simulate
from arboris.observers import PerfMonitor

from LQPctrl.task import FrameTask, MultiJointTask
from LQPctrl.task_ctrl import KpCtrl
from LQPctrl.LQPctrl import LQPcontroller

from numpy import pi, arange, array, zeros, eye

import pylab as pl

############### INIT ###################
gpos = [-1.,-.5,1.]
goal = {'inside': [.5, .5, 0], 'limit': [.78, .78, 0], 'beyond': [.95, .95, 0]}
gforcemax = {"Shoulder":10,"Elbow":5,"Wrist":2}

options = []
for cost in ['wrench consistent', 'dtwist consistent', 'a/m']:
    for norm in ['normal', 'inv(lambda)', 'inv(ellipsoid)']:
        for form in ['chi']: #'dgvel chi', 
            for g in ['inside','limit','beyond']:
                options.append(cost+"_"+norm+"_"+form+"_"+g)


############### SIMULATIONS ###################
jtraj ={}
traj = {}
results = {}

for opt in options:
    print """===============================================
START : {0}
===============================================
    """.format(opt)

    c,n,f,g = opt.split("_")
    w = create_3r_and_init(gpos=gpos)
    frames = w.getframes()
    joints = w.getjoints()

    tasks = []
    tasks.append(MultiJointTask(joints, KpCtrl(gpos, 10), [], .001, 0, True))
    ee_goal = eye(4)
    ee_goal[0:3,3] = goal[g]
    tasks.append(FrameTask(frames["EndEffector"], KpCtrl(ee_goal, 10), [3,4], 1., 0, True))
    
    lqpc = LQPcontroller(gforcemax, tasks=tasks, options={'cost':c,"norm":n,"formalism":f}, solver_options={"show_progress":False})
    w.register(lqpc)

    obs = [PerfMonitor(True), RecordJointPosition(joints), RecordFramePosition(frames["EndEffector"])]
    simulate(w, arange(0,3.,0.01), obs)

    jtraj[opt] = obs[-2].get_record()
    traj[opt] = obs[-1].get_record()
    results[opt] = lqpc.get_performance()


############### RESULTS ###################
for g in ['inside','limit','beyond']:
    for c in ['wrench consistent', 'dtwist consistent', 'a/m']:
        pl.figure()
        for k in traj:
            if c == k[:len(c)] and g == k[-len(g):]:
                cc,n,f,gg = k.split("_")
                if n=='normal':
                    pl.subplot(321); pl.plot(traj[k]); pl.title('goal: '+g);pl.ylabel(n)
                    xlim = pl.xlim()
                    pl.plot(xlim, [goal[g][0], goal[g][0]], 'r:'); pl.plot(xlim, [goal[g][1], goal[g][1]], 'r:')
                    pl.subplot(322); pl.plot(jtraj[k]); pl.title('cost: '+c)
                if n=='inv(lambda)':
                    pl.subplot(323); pl.plot(traj[k]); pl.ylabel(n)
                    xlim = pl.xlim()
                    pl.plot(xlim, [goal[g][0], goal[g][0]], 'r:'); pl.plot(xlim, [goal[g][1], goal[g][1]], 'r:')
                    pl.subplot(324); pl.plot(jtraj[k])
                if n=='inv(ellipsoid)':
                    pl.subplot(325); pl.plot(traj[k]); pl.ylabel(n); pl.xlabel('end effector')
                    xlim = pl.xlim()
                    pl.plot(xlim, [goal[g][0], goal[g][0]], 'r:'); pl.plot(xlim, [goal[g][1], goal[g][1]], 'r:')
                    pl.subplot(326); pl.plot(jtraj[k]); pl.xlabel('joints')
         # IF WE WANT TO SAVE THE FIG
#        ccc ="am" if c=='a/m' else c
#        pl.savefig('goal_'+g+'_cost_'+ccc+'.pdf', bbox_inches='tight')


pl.show()
