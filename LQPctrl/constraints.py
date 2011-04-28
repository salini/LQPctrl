#coding=utf-8
#author=Joseph Salini
#date=21 april 2011

from numpy import zeros, array, arange, dot, hstack, vstack

def eq_motion(M, JcT_S, gravity_N):
    """ equation of motion: M.dgvel + N = S.gforce + Jc_T.fc + gravity

    Rewrite:             |dgvel |
                         |fc    |
           [M, -Jc_T, -S]|gforce| = gravity - N
    """
    A = hstack( (M, -JcT_S) )
    b = gravity_N
    return A, b


def eq_contact_acc(Jc, dJc_gvel, n_problem, const_activity, formalism='dgvel chi', Minv_JcT_S=None, Minv_Grav_N=None):
    """ equality of contact acceleration: Jc.dgvel + dJc.gvel = 0

    for formalism 'dgvel chi':
    Rewrite:         |dgvel |
                     |fc    |
           [Jc, 0, 0]|gforce| = |-dJc.gvel|

    or for formalism 'chi':
    Rewrite:                |fc    |
           [Jc]Minv[Jc_T, S]|gforce| = |-dJc.gvel| - [Jc]Minv[G-N]
    """
    if len(Jc) == 0:
        A = zeros((0, n_problem))
        b = zeros(0)
    else:
        selected_dof = []
        selected_fc  = []
        for i in arange(len(const_activity)):
            if const_activity[i] is True:
                selected_dof.extend( arange(3*i, 3*(i+1)) )
            else:
                selected_fc.extend( arange(3*i, 3*(i+1)) )

        Adgvel = Jc[selected_dof, :]
        b = -dJc_gvel[selected_dof]

        A_fc = zeros((len(selected_fc), A.shape[1]))
        b_fc = zeros(len(selected_fc))
        if formalism == 'dgvel chi':
            A = zeros((len(selected_dof), n_problem))
            A[:Adgvel.shape[0],:Adgvel.shape[1]] = Adgvel
            A_fc[arange(len(selected_fc)), Jc.shape[0]+ array(selected_fc)] = 1
        elif formalism == 'chi':
            A = dot(Adgvel, Minv_JcT_S)
            b = b - hstack(Adgvel, Minv_Grav_N)
            A_fc[arange(len(selected_fc)), selected_fc] = 1

        A = vstack((A,A_fc))
        b = hstack((b,b_fc))

    return A, b



def ineq_gforcemax(gforcemax, n_dof, n_fc, formalism='dgvel chi'):
    """ inequality of gforcemax: - gforcemax <= gforce <= gforcemax
    
    for formalism 'dgvel chi':
    Rewrite:         |dgvel |
                     |fc    |
           [0, 0,  I]|gforce|    |gforcemax|
           [0, 0, -I]         <= |gforcemax|

    or for formalism 'chi':
    Rewrite:      |fc    |
           [0,  I]|gforce|    |gforcemax|
           [0, -I]         <= |gforcemax|
    """
    n_gforce = len(gforcemax)
    if formalism == 'dgvel chi':
        n_start = n_dof + n_fc
    elif formalism == 'chi':
        n_start = n_fc

    n_problem = n_start+n_gforce
    G = zeros((2*n_gforce, n_problem))
    G[arange(n_gforce), arange(n_start, n_problem)] = 1
    G[arange(n_gforce, 2*n_gforce), arange(n_start, n_problem)] = -1
    h = hstack( (gforcemax, gforcemax) )
    return G, h


def ineq_friction(mus, const_activity, n_pan, n_dof, n_problem, formalism='dgvel chi'):
    """ inequality for linearized friction cone: C.fc <= 0

    for formalism 'dgvel chi':
    Rewrite:         |dgvel |
                     |fc    |
                     |gforce|
           [0, C, 0]         <= |0|

    or for formalism 'chi':
    Rewrite:     |fc    |
                 |gforce|
           [C, 0]         <= |0|
    """
    if len(mus)==0:
        G = zeros((0, n_problem))
        h = zeros(0)
    else:
        from numpy import cos, sin
        c_theta_2 = cos(pi/n_pan)
        def _gen_cone(_mu):
            _mu2 = _mu * c_theta_2
            a = arange(n_pan)*2.*pi/n_pan
            gen_cone = zeros((n_pan, 3))
            gen_cone[:,0] = cos(a)
            gen_cone[:,1] = sin(a)
            gen_cone[:,2] = _mu2
            return gen_cone

        selected_fc  = []
        for i in arange(len(const_activity)):
            if const_activity[i] is True and mus[i] is not None:
                selected_fc.append(i)

        n_sel_fc = len(selected_fc)
        G = zeros((n_sel_fc, n_problem))
        h = zeros(n_sel_fc)
        if formalism == 'dgvel chi':
            n_start = n_dof
        elif formalism == 'chi':
            n_start = 0
        k=0
        for i in selected_fc:
            G[(k*n_pan):((k+1)*n_pan), (n_start+3*i):(n_start+3*(i+1))] = _gen_cone(mus[i])
            k+=1

    return G, h



def ineq_joint_limits(qlim, vlim, gpos, gvel, hpos, hvel, n_problem, formalism='dgvel chi', Minv_JcT_S=None, Minv_Grav_N=None):
    """ inequality of joint limits: B_min <= K.dgvel <= B_max
    with B_min = max(2*(pos_lim_dn - pos -hpos.gvel)/hpos**2, (vel_lim_dn - gvel)/hvel)
         B_max = min(2*(pos_lim_up - pos -hpos.gvel)/hpos**2, (vel_lim_up - gvel)/hvel)

    for formalism 'dgvel chi':
    Rewrite:         |dgvel |
                     |fc    |
           [ K, 0, 0]|gforce|    | B_max|
           [-K, 0, 0]         <= |-B_min|

    or for formalism 'chi':
    Rewrite:                |fc    |
           [ K]Minv[Jc_T, S]|gforce|    | B_max| - [ K]Minv[G-N]
           [-K]                      <= |-B_min|   [-K]
    """
    from numpy import fmin, fmax, nan, isnan

    B_min = fmax((vlim[:,0] - gvel)/hvel, 2*(qlim[:,0] - gpos - hpos*gvel)/hpos**2)
    B_max = fmin((vlim[:,1] - gvel)/hvel, 2*(qlim[:,1] - gpos - hpos*gvel)/hpos**2)

    selected_dof = [i for i in arange(len(B_min)) if not isnan(B_min[i])]
    n_lim_dof = len(selected_dof)

    K = zeros((2*n_lim_dof, len(B_min)))
    K[arange(n_lim_dof), selected_dof] = 1
    K[arange(n_lim_dof, 2*n_lim_dof), selected_dof] = -1

    h = hstack((B_max[selected_dof], -B_min[selected_dof]))

    if formalism == 'dgvel chi':
        G = zeros((2*n_lim_dof, n_problem))
        G[:, :len(B_min)] = K
    elif formalism == 'chi':
        G = dot(K, Minv_JcT_S)
        h = h - dot(K, Minv_Grav_N)

    return G, h




