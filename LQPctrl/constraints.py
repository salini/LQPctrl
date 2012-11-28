#coding=utf-8
#author=Joseph Salini
#date=21 april 2011

""" Set of all (eqaulity & inequality) constraints use in LQPctrl.

Each constraint is described by the couples:
    (A,b) for eqality constraints such as Ax = b
    (G,h) for ineqality constraints such as Gx <= h
"""
from numpy import zeros, array, arange, dot, hstack, vstack, fmin, fmax, \
                  isnan, cos, sin, pi, nan, isfinite, isinf

def eq_motion(M, Jchi_T, G_N):
    """ equation of motion: M.dgvel + N = S.gforce + Jc_T.fc + G

    Rewrite:             |dgvel |
                         |fc    |
           [M, -Jc_T, -S]|gforce| = G - N
    """
    A = hstack( (M, -Jchi_T) )
    b = G_N
    return A, b


def eq_contact_acc(Jc, dJc_gvel, n_problem, const_activity, \
                   formalism='dgvel chi', Minv_Jchi_T=None, Minv_G_N=None):
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

        A_fc = zeros((len(selected_fc), n_problem))
        b_fc = zeros(len(selected_fc))
        if formalism == 'dgvel chi':
            A = zeros((len(selected_dof), n_problem))
            A[:Adgvel.shape[0], :Adgvel.shape[1]] = Adgvel
            if len(selected_fc):
                A_fc[arange(len(selected_fc)), \
                     Jc.shape[1] + array(selected_fc)] = 1
        elif formalism == 'chi':
            A = dot(Adgvel, Minv_Jchi_T)
            b = b - dot(Adgvel, Minv_G_N)
            if len(selected_fc):
                A_fc[arange(len(selected_fc)), selected_fc] = 1

        A = vstack((A, A_fc))
        b = hstack((b, b_fc))

    return A, b



def ineq_gforcemax(gforcemax, dgforcemax, dt, gforce_prec, \
                   n_dof, n_fc, formalism='dgvel chi'):
    """ inequality of gforcemax: B_min <= gforce <= B_max
    with B_min = max(-gforcemax, gforce_prec - dgforcemax*dt)
         B_max = min( gforcemax, gforce_prec + dgforcemax*dt)

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
    if dgforcemax is not None:
        B_min = fmax(-gforcemax, gforce_prec - dgforcemax*dt)
        B_max = fmin( gforcemax, gforce_prec + dgforcemax*dt)
    else:
        B_min = -gforcemax
        B_max =  gforcemax

    n_gforce = len(gforcemax)
    if formalism == 'dgvel chi':
        n_start = n_dof + n_fc
    elif formalism == 'chi':
        n_start = n_fc

    n_problem = n_start+n_gforce
    G = zeros((2*n_gforce, n_problem))
    G[arange(n_gforce), arange(n_start, n_problem)] = 1
    G[arange(n_gforce, 2*n_gforce), arange(n_start, n_problem)] = -1
    h = hstack( (B_max, -B_min) )
    return G, h


def ineq_friction(mus, const_activity, n_pan, n_dof, \
                  n_problem, formalism='dgvel chi'):
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
        c_theta_2 = cos(pi/n_pan)
        def _gen_cone(_mu):
            """ generic cone
            """
            _mu2 = _mu * c_theta_2
            a = arange(n_pan)*2.*pi/n_pan
            gen_cone = zeros((n_pan, 3))
            gen_cone[:, 0] = cos(a)
            gen_cone[:, 1] = sin(a)
            gen_cone[:, 2] = - _mu2
            return gen_cone

        selected_fc  = []
        for i in arange(len(const_activity)):
            if const_activity[i] is True and mus[i] is not None:
                selected_fc.append(i)

        n_sel_fc = n_pan*len(selected_fc)
        G = zeros((n_sel_fc, n_problem))
        h = zeros(n_sel_fc)
        if formalism == 'dgvel chi':
            n_start = n_dof
        elif formalism == 'chi':
            n_start = 0
        k = 0
        for i in selected_fc:
            G[(k*n_pan):((k+1)*n_pan), \
              (n_start+3*i):(n_start+3*(i+1))] = _gen_cone(mus[i])
            k += 1

    return G, h




def ineq_joint_limits(qlim, vlim, gpos, gvel, hpos, hvel, n_problem, \
                      formalism='dgvel chi', Minv_Jchi_T=None, Minv_G_N=None):
    """ inequality of joint limits: B_min <= K.dgvel <= B_max
    with
    B_min = max(2*(pos_lim_dn - pos -hpos.gvel)/hpos**2, (-vel_max - gvel)/hvel, gvel**2/(2*(gpos - pos_lim_dn)).Cond(tmin>0) )
    B_max = min(2*(pos_lim_up - pos -hpos.gvel)/hpos**2, ( vel_max - gvel)/hvel, gvel**2/(2*(gpos - pos_lim_up)).Cond(tmax>0) )

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
    
    ##### Compute bounds for velocity limits considering horizon position #####
    bmin_vel = (-vlim - gvel)/hvel
    bmax_vel = ( vlim - gvel)/hvel
    
    ##### Compute bounds for position limits considering horizon position #####
    bmin_pos = 2*(qlim[:, 0] - gpos - hpos*gvel)/hpos**2
    bmax_pos = 2*(qlim[:, 1] - gpos - hpos*gvel)/hpos**2
    
    ##### Compute bounds for position limits considering time of inflexion #####
    tmin = -2*(gpos - qlim[:, 0]) / gvel
    tmax = -2*(gpos - qlim[:, 1]) / gvel

    tmin[tmin <=  0  ] = nan
    tmin[tmin > hpos] = nan
    tmin[isfinite(tmin)] = 1.
    tmax[tmax <=  0  ] = nan
    tmax[tmax > hpos] = nan
    tmax[isfinite(tmax)] = 1.

    bmin_pos_time = gvel**2/(2*(gpos - qlim[:, 0])) * tmin
    bmax_pos_time = gvel**2/(2*(gpos - qlim[:, 1])) * tmax

    ##### Concatenate bounds #####
    B_min = bmin_vel
    B_max = bmax_vel
    B_min = fmax(B_min, bmin_pos)
    B_max = fmin(B_max, bmax_pos)
    B_min = fmax(B_min, bmin_pos_time)
    B_max = fmin(B_max, bmax_pos_time)

    ##### Create K, concatenation of B_max & B_min #####
    smax_dof = [i for i in arange(len(B_max)) if not isnan(B_max[i])]
    smin_dof = [i for i in arange(len(B_min)) if not isnan(B_min[i])]
    
    nmax = len(smax_dof)
    nmin = len(smin_dof)
    K = zeros((nmax+nmin, len(B_min)))
    
    K[arange(nmax),            smax_dof] =  1
    K[arange(nmax, nmax+nmin), smin_dof] = -1

    ##### Create G & h, depending on formalism #####
    h = zeros(nmax+nmin)
    h[:nmax] =  B_max[smax_dof]
    h[nmax:] = -B_min[smin_dof]
    
    if formalism == 'dgvel chi':
        G = zeros((len(K), n_problem))
        G[:, :len(B_min)] = K
    elif formalism == 'chi':
        G = dot(K, Minv_Jchi_T)
        h = h - dot(K, Minv_G_N)

    return G, h



def ineq_collision_avoidance(sdist, svel, J, dJ, gvel, \
                             hpos, n_problem, formalism='dgvel chi', \
                             Minv_Jchi_T=None, Minv_G_N=None):
    """ inequality collision avoidance: K.dgvel <= B
    with K = -J and B = 2/hpos**2 * (sdist + hpos*svel) + dJ.gvel

    for formalism 'dgvel chi':
    Rewrite:         |dgvel |
                     |fc    |
           [ K, 0, 0]|gforce| <= | B |

    or for formalism 'chi':
    Rewrite:                |fc    |
           [ K]Minv[Jc_T, S]|gforce| <= | B | - [ K]Minv[G-N]
    """
    #TODO: complete this function to take into account time of inflexion
    B = 2*(sdist + hpos*svel)/hpos**2 + dot(dJ, gvel)
    K = -J

    h = B
    if formalism == 'dgvel chi':
        G = zeros((K.shape[0], n_problem))
        G[:, :K.shape[1]] = K
    elif formalism == 'chi':
        G = dot(K, Minv_Jchi_T)
        h = h - dot(K, Minv_G_N)

    return G, h



