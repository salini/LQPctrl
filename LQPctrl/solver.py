#coding=utf-8
#author=Joseph Salini
#date=21 april 2011


from numpy import array, dot, sum, where, diag, eye


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
    from numpy.linalg import svd, LinAlgError
    try:
        u,s,vh = svd(A, full_matrices=False)
        r = sum(where(s>1e-3, 1, 0)) # compute the rank of A
        ur,sr,vhr = u[:, :r], s[:r], vh[:r, :]
        Ar = dot(diag(sr), vhr)
        br = dot(ur.T, b)
    except (LinAlgError):
        Ar= A.copy()
        br = b.copy()
    return Ar, br

def init_solver(solver='cvxopt', options=None):
    """
    """
    if solver is 'cvxopt':
        from cvxopt import solvers
        if options is not None:
            solvers.options.update(options)
    else:
        raise ValueError, 'The required solver is not implemented. try another solver.'


def solve(E, f, G, h, A, b, solver='cvxopt'):
    """
    """
    

    A, b = _reduce_constraints(A, b)
    if solver is 'cvxopt':
        X_solution = _solve_cvxopt(E, f, G, h, A, b)
    else:
        raise ValueError, 'The required solver is not implemented. try another solver.'

    return X_solution


def _solve_cvxopt(E, f, G, h, A, b):
    """
    """
    from cvxopt import matrix
    from cvxopt.solvers import qp as qpsolver

    P = 2*dot(E.T, E)
    q = 2*dot(f, E)
    Pp, qp, Gp, hp, Ap, bp = matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b)

    degenerate_rank = -16
    for i in range(10):
        try:
            results = qpsolver(Pp, qp, Gp, hp, Ap, bp)
        except ValueError as err:
            print "Exception Error:",err.args
            if err.args[0] == "Rank(A) < p or Rank([P; A; G]) < n":
                print 'Try to degenerate P to ensure rank([P; A; G])=n'
                P += eye(P.shape[0])*10**(degenerate_rank)
                Pp = matrix(P)
                degenerate_rank += 1
            elif err.args[0] == "domain error":
                print "Should delete some constraints"
        else:
            break

    X_sol = array(results['x']).flatten()
    return X_sol
