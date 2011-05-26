#!/usr/bin/python
#coding=utf-8
#author=Joseph Salini
#date=16 may 2011

from common import create_3r_and_init, print_lqp_perf, RecordFramePosition

from arboris.core import simulate
from arboris.homogeneousmatrix import rotz
from arboris.observers import PerfMonitor

from LQPctrl.task import FrameTask
from LQPctrl.task_ctrl import KpCtrl
from LQPctrl.LQPctrl import LQPcontroller

from numpy import pi, arange, array, zeros

import pylab as pl

############### INIT ###################
goal = rotz(pi/8)
goal[0:3,3] = [-0.4, .5, 0]
gforcemax = {"Shoulder":10,"Elbow":5,"Wrist":2}

options = []
for cost in ['normal', 'wrench consistent']:
    for norm in ['normal', 'inv(lambda)']:
        for form in ['dgvel chi', 'chi']:
            options.append(cost+";"+norm+";"+form)


############### SIMULATIONS ###################
results = {}
traj = {}
for opt in options:
    print """===============================================
START : {0}
===============================================
    """.format(opt)

    w = create_3r_and_init(gpos=(.5,.5,.5))
    frames = w.getframes()

    tasks = []
    tasks.append(FrameTask(frames["EndEffector"], KpCtrl(goal, 20), [], 1., 0, True))
    c,n,f = opt.split(";")
    lqpc = LQPcontroller(gforcemax, tasks=tasks, options={'cost':c,"norm":n,"formalism":f}, solver_options={"show_progress":False})
    w.register(lqpc)

    obs = [PerfMonitor(True), RecordFramePosition(frames["EndEffector"])]
    simulate(w, arange(0,3.,0.01), obs)

    traj[opt] = obs[-1].get_record()
    results[opt] = lqpc.get_performance()


############### RESULTS ###################
pl.figure()
for k in traj:
    pl.plot(traj[k])
pl.title("trajectories of end effector")


label = [('update robot state', 'up rstate', "g"), ('update tasks and events', 'up tasks','b'),
         ('get constraints', 'get const','c'), ('sort tasks', 'sort tasks','m'), ('get cost function', 'get cost', 'y'),
         ('solve', 'solve','r'), ('constrain next level', 'const next', 'k')]
ind = arange(len(options))
starting = zeros(len(options))
width = .5
lines = []
pl.figure()
for l,s,c in label:
    high = array([results[n][l] for n in options])*1000.
    lines.append( pl.bar(ind, high, width, starting, color=c, label=s) )
    starting += high

pl.ylabel('time (ms)')
pl.title('Performance of LQPcontroller with different options')
pl.xticks(ind+width/2., ["".join([v[0] for v in k.split(";")]) for k in options] )
pl.axes().set_xlim(0,len(options)+3)

handles, labels = pl.axes().get_legend_handles_labels()
pl.axes().legend(handles[::-1], labels[::-1], loc='lower right')
pl.gca().yaxis.grid(True)



pl.show()
