#coding=utf-8
#author=Joseph Salini
#date=20 may 2011
"""
date 10th march 2010
@author: Joseph Salini

This module regroup several useful functions
"""


from numpy import array, zeros, sqrt, sign, dot, cross
""" QUATERNIONS

This part of the module regroup the functions to quaternion calculus
"""
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
