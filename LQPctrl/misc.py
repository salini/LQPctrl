#coding=utf-8
#author=Joseph Salini
#date=20 may 2011


from numpy import array, zeros, sqrt, sign, dot, cross
from numpy.linalg import norm

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

    def __add__(self, obj):
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
        """make the substraction between the quatpos instance and an other (expressed in the same frame)

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
    from arboris.homogeneousmatrix import inv, iadjoint, dAdjoint
    from arboris.massmatrix import principalframe

    H_body_com = principalframe(body.mass)
    H_0_com = dot(body.pose, H_body_com)
    P_0_com = H_0_com[0:3, 3]

    if compute_J:
        H_com_com2 = inv(H_0_com)
        H_com_com2[0:3,3] = 0.
        Ad_com2_body = iadjoint(dot(H_body_com, H_com_com2))
        J_com2 = dot(Ad_com2_body, body.jacobian)

        T_com2_body = body.twist.copy()
        T_com2_body[3:6] = 0.
        dAd_com2_body = dAdjoint(Ad_com2_body, T_com2_body)
        dJ_com2 = dot(Ad_com2_body, body.djacobian) + dot(dAd_com2_body, body.jacobian)
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
        m = b.mass[3,3]
        if m >=1e-10:
            mass_sum += m
            result = body_com_properties(b, compute_J)
            if compute_J:
                P_com, J_com, dJ_com = result
                P_com_sum += m*P_com
                J_com_sum += m*J_com[3:6,:]
                dJ_com_sum = m*dJ_com[3:6,:]
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
    from arboris.massmatrix import principalframe, transport
    R0_sum = zeros(3)
    M0_sum = zeros(3)
    for b in bodies:
        Mbody = b.mass
        m = Mbody[3,3]
        if m >=1e-10:
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
    zmp = cross(n,M0_sum) / dot(R0_sum,n)
    return zmp
