#coding=utf-8
#author=Joseph Salini
#date=20 may 2011

from arboris.homogeneousmatrix import inv, iadjoint, dAdjoint
from arboris.massmatrix import principalframe, transport

from numpy import array, zeros, sqrt, sign, dot, cross
from numpy.linalg import norm

from scipy.interpolate import piecewise_polynomial_interpolate as ppi
from numpy import arange, arctan2, sin, eye, log, exp

################################################################################
#
# QUATERNIONS
#
################################################################################
def quat2rot(real, img):
    """Convert a normalized quaternion to a rotation matrix

    inputs:
    real: a scalar which represents the real part of the quaternion
    img: a vector(3) which represent the imaginary part of the quaternion

    return a 3x3 rotation matrix
    """
    n = real
    q0, q1, q2 = img
    return 2*array([[n**2+q0**2 -.5, q0*q1-n*q2    , q0*q2+n*q1    ],
                    [q0*q1+n*q2    , n**2+q1**2 -.5, q1*q2-n*q0    ],
                    [q0*q2-n*q1    , q1*q2+n*q0    , n**2+q2**2 -.5]])

def rot2quat(R):
    """Convert a rotation matrix to a normalized quaternion

    inputs:
    R: a 3x3 rotation matrix

    return a tuple (real, img) the representation of a the quaternion
    real is a scalar and img is a vector(3)
    the quaternion is normalized
    """
    real = sqrt(R[0][0]+R[1][1]+R[2][2]+1.)/2.
    s = [sign(R[2][1]-R[1][2]), sign(R[0][2]-R[2][0]), sign(R[1][0]-R[0][1])]
    for i in range(len(s)):
        if s[i] == 0:
            s[i] = 1
    img = array([s[0]*sqrt(abs( R[0][0]-R[1][1]-R[2][2]+1.))/2.,
                 s[1]*sqrt(abs(-R[0][0]+R[1][1]-R[2][2]+1.))/2.,
                 s[2]*sqrt(abs(-R[0][0]-R[1][1]+R[2][2]+1.))/2.])
    return real, img

class quaternion:
    """ The class for the quaternion

    functions:
    normalize: normalize the quaternion
    norm: give the norm of the quaternion
    to_rot: convert the quaternion into a 3x3 rotation matrix
    C: give the conjugate of the instance quaternion: q(real, -img)
    inv: give the inverse of the instance quaternion: q.C/q.norm()
    """
    def __init__(self, real=None, img=None, R=None):
        """ Initialize the quaternion

        inputs:
        R (None): instanciate from a 3x3 rotation matrix
        real (None): give the real part of the quaternion
        img (None): give the imaginary part of the quaternion

        If no input is given, return null quaternion: [0,(0,0,0)]
        """
        self.real = 0.
        self.img = zeros(3)
        if R is not None:
            self.real, self.img = rot2quat(R)
        if real is not None:
            self.real = real
        if img is not None:
            assert(len(img) == 3)
            self.img[:] = img

    def normalize(self):
        """ Return a new instance of the Normalized quaternion
        """
        n = self.norm()
        return quaternion(real=self.real/n, img=self.img/n)

    def norm(self):
        """ Return the norm of the instance quaternion
        """
        return sqrt(self.real**2 + dot(self.img, self.img))

    def to_rot(self):
        """ Convert a quaternion in a 3x3 rotation matrix
        """
        # we do NOT change the instance, but we need normalized
        # quaternion to convert into rotation matrix
        n = self.norm()
        return quat2rot(self.real/n, self.img/n)

    def to_array(self):
        """ Convert the quaternion in a array of 4 dimensions [real, img]
        """
        return array([self.real, self.img[0], self.img[1], self.img[2]])

    @property
    def C(self):
        """ Return the conjugate quaternion
        """
        return quaternion(real=self.real, img=-self.img)

    @property
    def inv(self):
        """ Return the inverse of the quaternion
        """
        n = self.real**2 + dot(self.img, self.img)
        return quaternion(real=self.real/n, img=-self.img/n)

    def copy(self):
        """ Return a copy of the quaternion
        """
        return quaternion(real=self.real, img=self.img)

    def __add__(self, Q):
        """ The addition between self and an other quaternion
        """
        assert (isinstance(Q, quaternion))
        real = self.real + Q.real
        img = self.img + Q.img
        return quaternion(real=real, img=img)

    def __sub__(self, Q):
        """ The substraction between self and an other quaternion
        """
        assert (isinstance(Q, quaternion))
        real = self.real - Q.real
        img = self.img - Q.img
        return quaternion(real=real, img=img)

    def __neg__(self):
        """ Return the negative of the quaternion
        """
        return quaternion(real=-self.real, img=-self.img)

    def __mul__(self, Q):
        """ Return the multiplication with 2 quaternions
        """
        real = self.real*Q.real - dot(self.img, Q.img)
        img = self.real*Q.img + Q.real*self.img + cross(self.img, Q.img)
        return quaternion(real=real, img=img)

    def __repr__(self):
        """represention of a quaternion
        """
        return '[{0},({1}, {2}, {3})]'.format(
               self.real, self.img[0], self.img[1], self.img[2])


class quatpos:
    """A class which represents the concatenation of a position and a quaternion
    """
    def __init__(self, H):
        self.R = H[0:3, 0:3]
        self.Q = quaternion(R = self.R)
        self.p = H[0:3, 3]

    def to_htr(self):
        """Convert the quatpos instance in a 4x4 homogeneous matrix
        """
        H = eye(4)
        H[0:3, 0:3] = self.Q.to_rot()
        H[0:3, 3]   = self.p
        return H

    def __sub__(self, qv):
        """make the substraction between the quatpos instance
        and an other (expressed in the same frame)

        the difference is the concatenation of 2 differences:
        the 3 first are the rotational difference
        the 3 last are the position difference

        the result is expressed in the frame where are expressed the 2 quatpos
        """
        assert(isinstance(qv, quatpos))
        res = zeros(6)
        Q_res = qv.Q.inv*self.Q
        #res[0:3] = sign(Q_res.real)*Q_res.img
        #res[3:6] = dot(qv.R.T, self.p - qv.p)
        res[0:3] = sign(Q_res.real)*dot(qv.R, Q_res.img)
        res[3:6] = self.p - qv.p
        return res

    def __repr__(self):
        """representation of a quatpos instance
        """
        return 'quat: [{0},({1}, {2}, {3})] pos: ({4}, {5}, {6})'.format(
                self.Q.real, self.Q.img[0], self.Q.img[1], self.Q.img[2],
                self.p[0], self.p[1], self.p[2])




################################################################################
#
# COM PROPERTIES COMPUTATION
#
################################################################################
def body_com_properties(body, compute_J=True):
    """ Compute the Center of Mass properties of a body.
    """

    H_body_com = principalframe(body.mass)
    H_0_com = dot(body.pose, H_body_com)
    P_0_com = H_0_com[0:3, 3]

    if compute_J:
        H_com_com2 = inv(H_0_com)
        H_com_com2[0:3, 3] = 0.
        Ad_com2_body = iadjoint(dot(H_body_com, H_com_com2))
        J_com2 = dot(Ad_com2_body, body.jacobian)

        T_com2_body = body.twist.copy()
        T_com2_body[3:6] = 0.
        dAd_com2_body = dAdjoint(Ad_com2_body, T_com2_body)
        dJ_com2 = dot(Ad_com2_body, body.djacobian) + \
                  dot(dAd_com2_body, body.jacobian)
        return P_0_com, J_com2, dJ_com2
    else:
        return P_0_com


def com_properties(bodies, compute_J=True):
    """ Compute the Center of Mass properties of many bodies.
    """
    mass_sum = 0.
    P_com_sum = 0.
    J_com_sum = 0.
    dJ_com_sum = 0.
    for b in bodies:
        m = b.mass[3, 3]
        if m >= 1e-10:
            mass_sum += m
            result = body_com_properties(b, compute_J)
            if compute_J:
                P_com, J_com, dJ_com = result
                P_com_sum += m*P_com
                J_com_sum += m*J_com[3:6, :]
                dJ_com_sum = m*dJ_com[3:6, :]
            else:
                P_com = result
                P_com_sum += m*P_com

    P_com = P_com_sum/mass_sum
    J_com = J_com_sum/mass_sum
    dJ_com = dJ_com_sum/mass_sum

    if compute_J:
        return P_com, J_com, dJ_com
    else:
        return P_com


def zmp_position(bodies, g, gvel, dgvel, n=None):
    """
    """
    R0_sum = zeros(3)
    M0_sum = zeros(3)
    for b in bodies:
        Mbody = b.mass
        m = Mbody[3, 3]
        if m >= 1e-10:
            H_body_com = principalframe(Mbody)
            Mcom = transport(Mbody, H_body_com)
            I = Mcom[0:3, 0:3]
            P_com, J_com, dJ_com = body_com_properties(b)
            twist = dot(J_com, gvel)
            dtwist = dot(J_com, dgvel) + dot(dJ_com, gvel)

            R0 = dot(m, (dtwist[3:6] - g))
            M0 = dot(m, cross(P_com, (dtwist[3:6] - g))) + \
                 (dot(I, dtwist[0:3]) - cross(dot(I, twist[0:3]), twist[0:3]))
            R0_sum += R0
            M0_sum += M0

    if n is None:
        n = g/norm(g)
    zmp = cross(n, M0_sum) / dot(R0_sum, n)
    return zmp




################################################################################
#
# CONVEX HULL COMPUTATION
#
################################################################################
def convex_hull(point):
    """
    """
    def on_left(p0, p1, p2):
        """
        P2 is left of P0->P1 if result > 0
        P2 is on P0->P1 line if result = 0
        P2 is right of P0->P1 if result < 0
        in Matlab:
        res = (P1(1) - P0(1))*(P2(2) - P0(2)) - (P2(1) - P0(1))*(P1(2) - P0(2))
        """
        res = (p1[0]-p0[0])*(p2[1]-p0[1])  -  (p2[0]-p0[0])*(p1[1]-p0[1])
        if res > 0:
            return True
        else:
            return False

    valid_pt = list(point)
    nb_pt = len(valid_pt)
    if nb_pt in [0, 1]:
        return []
    elif nb_pt == 2:
        return [valid_pt[i] for i in [0, 1, 0]]

    CH = []
    #selection of the first point
    selected_pt = valid_pt[0]
    for p in valid_pt:
        if p[0] < selected_pt[0]:
            selected_pt = p
    CH.append(selected_pt)
    for i in range(len(valid_pt)):
        if valid_pt[i] is selected_pt:
            valid_pt.pop(i)
            break

    while len(valid_pt)>0:
        #We select the last point on the CH list, and we test
        #with the next valid points
        selected_pt = valid_pt[0]
        for pt in valid_pt[1:]:
            if on_left(CH[-1], selected_pt, pt):
                selected_pt = pt

        if on_left(CH[-1], selected_pt, CH[0]):
            CH.append(CH[0])
            break
        else:
            CH.append(selected_pt)
            for i in range(len(valid_pt)):
                if valid_pt[i] is selected_pt:
                    valid_pt.pop(i)
                    break

    if len(valid_pt) == 0:
        CH.append(CH[0])

    return CH


def is_in_convex_hull(CH, point, margin=0.):
    """ WARNING: CHECK VALIDITY WITH MARGIN!!!
    """
    if len(CH) <= 2:
        return False

    is_in = True
    for i in range(len(CH)-1):
        n = array([ float(CH[i+1][1]-CH[i][1]), -float((CH[i+1][0]-CH[i][0]))])
        n /= norm(n)
        ch0 = CH[i]   + margin*n
        ch1 = CH[i+1] + margin*n
        if ((ch1[0] - ch0[0])*(point[1] - ch0[1]) - \
            (point[0] - ch0[0])*(ch1[1] - ch0[1])) > 0:
            is_in = False
            break
    return is_in


def extract_contact_points(poses, dof):
    """
    We extract the x and z components of the projected point
    in the Rfloor coordinate
    """
    point = []
    for p in poses:
        point.append(p.pose[dof, 3])
    return point




################################################################################
#
# MISC COMPUTATION
#
################################################################################
def interpolate_log(start, end, tend, dt):
    logs, loge = log(start), log(end)
    logtrans = ppi([0, tend], [[logs, 0], [loge, 0]], \
                   arange(0, round(tend/dt + 1))*dt)
    return [exp(i) for i in logtrans]



def slerp(q0, q1, t, shortest_path=True):
    """ Compute Slerp (Spherical Linear Interpolation) between 2 quaternions.

    inputs:
    q0: The beginning quaternion, must be an instance quaternion
    q1: The ending quaternion, must be an instance quaternion
    t: a float between 0 and 1
    shortest_path (True): ensure to compute the quaternion on the shorthest path

    The goal is to find the quaternion as:
    q1 = q0*[cos(a), N.sin(a)]
    qt = q0*[cos(t*a), N.sin(t*a)]
    where a represents one angle (different from the angle of rotation)
    and N the normalized vector of 3 components
    q0,q1 and qt must be normalized, Hence:

    qt = ( sin(a*(1-t))*q0 + sin(a*t)*q1 )/sin(a)

    Returns qt
    """
    assert(isinstance(q0, quaternion))
    assert(isinstance(q1, quaternion))
    q0n = q0.normalize()
    q1n = q1.normalize()
    q0_1n = q0n.inv*q1n
    ca = q0_1n.real         #cos(a)
    sa = norm(q0_1n.img)    #sin(a)
    if ca < 0 and shortest_path:
        ca = -ca
        q1n = -q1n
    a = arctan2(sa, ca)
    if abs(sa) > 1e-8:
        q0coeff = sin(a*(1-t))/sin(a)
        q1coeff = sin(a*t)/sin(a)
        real = q0coeff*q0n.real + q1coeff*q1n.real
        img  = q0coeff*q0n.img  + q1coeff*q1n.img
        return quaternion(real=real, img=img)
    else:
        return q0n


def simple_der(y, dt):
    dy0 = (y[1]-y[0])/dt
    dyn = (y[-1]-y[-2])/dt
    Dt = 2*dt
    dy = [(y[i+1]-y[i-1])/Dt for i in range(1, len(y)-1)]
    return [dy0]+dy+[dyn]


def interpolate_rotation_matrix(R0, R1, tend, dt):
    """ Give the interpolation between 2 rotation matrices

    inputs:
    R0: the beginning rotation matrix
    R1: the ending rotation matrix
    tend: the end time
    dt: the time step

    this interpolation is done with no velocity or acceleration
    at the beginning and at the end

    Returns 3 lists: ([R], [omega], [domega])
    """
    def der_q(qim1, qip1, dt):
        dqi = qip1 - qim1
        return dqi.to_array()/dt

    def Q(q):
        r = q.real
        i0, i1, i2 = q.img
        return array([[-i0,   r, -i2,  i1],
                      [-i1,  i2,   r, -i0],
                      [-i2, -i1,  i0,   r]])

    q0, q1 = quaternion(R=R0), quaternion(R=R1)
    fraction = ppi([0, tend], [[0, 0, 0], [1, 0, 0]], arange(0, tend+dt, dt))
    q = [slerp(q0, q1, f) for f in fraction]
    R = [qi.to_rot() for qi in q]

    dq = [der_q(q[0], q[1], dt)] + \
          [der_q(q[i-1], q[i+1], 2*dt) for i in range(1, len(q)-1)] + \
          [der_q(q[-2], q[-1], dt)]

    omega = [2*dot(Q(q[i]), dq[i]) for i in range(len(q))]
    domega = simple_der(omega, dt)

    return R, omega, domega


def interpolate_vector(v0, v1, tend, dt):
    """ Give the interpolation between 2 vectors

    inputs:
    v0: the beginning rotation matrix
    v1: the ending rotation matrix
    tend: the end time
    dt: the time step

    this interpolation is done with no velocity or acceleration
    at the beginning and at the end

    Returns 3 lists: ([v], [dv], [ddv])
    """
    assert(len(v0)==len(v1))
    v = array([ppi([0, tend], [[v0[i], 0, 0], [v1[i], 0, 0]], \
               arange(0, tend+dt, dt)) for i in range(len(v0))]).T
    dv = simple_der(v, dt)
    ddv = simple_der(dv, dt)

    return v, dv, ddv


def interpolate_htr(H_begin, H_end, tend, dt):
    """Give the interpolation between 2 frames

    inputs:
    H_begin: the begin frame, a 4x4 homogeneous matrix
    H_end: the end frame, a 4x4 homogeneous matrix
    tend: the end time
    dt: the time step

    we interpolate from H_begin to H_end with velocities and
    accelerations null at the beginning and at the end

    Return 3 list: ([H],[twist],[dtwist])
    """
    def make_htr(R, p):
        H = eye(4)
        H[0:3, 0:3] = R
        H[0:3, 3] = p
        return H

    R, w, dw = interpolate_rotation_matrix(H_begin[0:3, 0:3], \
                                           H_end[0:3, 0:3], tend, dt)
    p, v, dv = interpolate_vector(H_begin[0:3, 3], H_end[0:3, 3], tend, dt)
    pos = [make_htr(R[i], p[i]) for i in range(len(R))]
    vel = [array([w[i], v[i]]).flatten() for i in range(len(w))]
    acc = [array([dw[i], dv[i]]).flatten() for i in range(len(dw))]
    return pos, vel, acc
