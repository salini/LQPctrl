#coding=utf-8
#author=Joseph Salini
#date=21 april 2011


from numpy import array, dot, sum as np_sum, where, diag, eye
from numpy.linalg import svd, LinAlgError

from cvxopt import solvers, matrix
from cvxopt.solvers import qp as qpsolver

def _reduce_constraints(A, b):
    """ Make the constraint non-singular

    if the constraint is on the form:
    dot(A,x) = b
    A may be singular. to avoid this problem, we extract the
    non-singular part of the equation thanks to svd:
    A = U*S*Vh with U.T*U = I and Vh.T*Vh = I
    if r is the rank of A, we have:
    Ar = S[:r,:r]*Vh[:r,:]
    br = U[:,:r].T*b
    Hence:
    Ar*x = br
    """
    
    try:
        u, s, vh = svd(A, full_matrices=False)
        r = np_sum(where(s>1e-3, 1, 0)) # compute the rank of A
        ur, sr, vhr = u[:, :r], s[:r], vh[:r, :]
        Ar = dot(diag(sr), vhr)
        br = dot(ur.T, b)
    except (LinAlgError):
        Ar = A.copy()
        br = b.copy()
    return Ar, br

def init_solver(solver='cvxopt', options=None):
    """
    """
    if solver is 'cvxopt':
        if options is not None:
            solvers.options.update(options)
    else:
        raise ValueError, \
              'The required solver is not implemented. try another solver.'


def solve(E, f, G, h, A, b, solver='cvxopt'):
    """
    """
    

    A, b = _reduce_constraints(A, b)
    if solver is 'cvxopt':
        X_solution = _solve_cvxopt(E, f, G, h, A, b)
    else:
        raise ValueError, \
              'The required solver is not implemented. try another solver.'

    return X_solution


def _solve_cvxopt(E, f, G, h, A, b):
    """
    """
    P = 2*dot(E.T, E)
    q = 2*dot(f, E)
    Pp = matrix(P)
    qp = matrix(q)
    Gp = matrix(G)
    hp = matrix(h)
    Ap = matrix(A)
    bp = matrix(b)

    for degenerate_rank in range(-16, -5):
        try:
            results = qpsolver(Pp, qp, Gp, hp, Ap, bp)
        except ValueError as err:
            print "Exception Error:", err.args
            if err.args[0] == "Rank(A) < p or Rank([P; A; G]) < n":
                print 'Try to degenerate P to ensure rank([P; A; G])=n'
                P += eye(P.shape[0])*10**(degenerate_rank)
                Pp = matrix(P)
            elif err.args[0] == "domain error":
                print "Should delete some constraints"
        else:
            break

    X_sol = array(results['x']).flatten()
    return X_sol
