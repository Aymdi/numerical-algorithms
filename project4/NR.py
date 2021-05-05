import numpy as np 

def compute_params(f, J, U0):
    """

    Compute shapes of parameters to addapt them to the Newton-Raphson algorithm

    Parameters
    ----------
    f: :class:`function`
        The function under study
    J: :class:`function`
        The Jacobian matrix associated to the function
    U0: Union[:class:`numpy.ndarray`, :class:`int`]
        The starting position of the algorithm

    Return
    ------
    :class:`function`
        The function under study with new shape for parameter and return
    :class:`function`
        The Jacobian matrix associated to the function with new shape for
        parameter and return
    :class:`numpy.ndarray`
        The starting position of the algorithm with new shape
    """

    m1 = np.array(U0)
    if len(m1.shape) == 0:

        def pr2pu(x):
            return x[0][0]

        _U0 = m1.reshape(1, 1)
        n = 1
    elif len(m1.shape) == 1:

        def pr2pu(x):
            return x.reshape(m1.shape)

        _U0 = m1.reshape(m1.shape[0], 1)
        n = m1.shape[0]
    elif len(m1.shape) == 2 and m1.shape[1] == 1:

        def pr2pu(x):
            return x

        _U0 = m1
        n = m1.shape[0]
    else:
        raise Exception(
            "Impossible to use Newton-Raphson method with starting point of shape {}"
            .format(m1.shape))
    m2 = np.array(f(U0))
    if len(m2.shape) == 0:

        def f2pr(x):
            return np.array(x).reshape(1, 1)

        m = 1
    elif len(m2.shape) == 1:

        def f2pr(x):
            return np.array(x).reshape(m2.shape[0], 1)

        m = m2.shape[0]
    elif len(m2.shape) == 2 and m2.shape[1] == 1:

        def f2pr(x):
            return x

        m = m2.shape[0]
    else:
        raise Exception(
            "Function f is supposed to return a vector, got matrice of shape {}"
            .format(m2.shape))
    m3 = np.array(J(U0))
    if m3.shape == (m, n):

        def j2pr(x):
            return x
    elif m3.shape == (n, m):

        def j2pr(x):
            return x.transpose()
    elif m3.shape == () and m == n == 1:

        def j2pr(x):
            return np.array(x).reshape(1, 1)
    elif m3.shape == (n) and m == 1:

        def j2pr(x):
            return np.array(x).reshape(1, n)
    elif m3.shape == (m) and n == 1:

        def j2pr(x):
            return np.array(x).reshape(m, 1)
    else:
        raise Exception(
            "Function f is from dimension {} to dimension {} so J is supposed to return a matrix of shape ({}), but is of shape ({})"
            .format(n, m, (n,m), m3.shape))
    return lambda x: f2pr(f(pr2pu(x))), lambda x: j2pr(J(pr2pu(x))), _U0, pr2pu


def Newton_Raphson(f, J, U0, epsilon, N=500, a_min=0.001):
    """

    Apply the Newton-Raphson method on a multi-dimensionnal array

    Parameters
    ----------
    f: :class:`function`
        The function under study
    J: :class:`function`
        The Jacobian matrix associated to the function
    U0: Union[:class:`numpy.ndarray`, :class:`int`]
        The starting position of the algorithm
    N: :class:`int`
        The maximal number of iterations allowed
    epsilon: :class:`int`
        The precision wanted

    Return
    ------
    :class:`numpy.ndarray`
        The solution of the problem
    """
    _f, _J, U, pr2pu = compute_params(f, J, U0)
    n = 0
    V = [epsilon + 1]
    fu = _f(U)
    while n < N and np.linalg.norm(fu) > epsilon:
        V = np.linalg.lstsq(_J(U), -fu, rcond=None)[0][0]
        a = 1
        fuav = _f(U + a * V)
        while np.linalg.norm(fuav) > np.linalg.norm(fu) and a > a_min:
            a /= 2
            fuav = _f(U + a * V)
        U = U + a * V
        fu = fuav
        n += 1
    return pr2pu(U)


####################################################################################


def test_Newton_Raphson():

    # dimension 1*1
    def f1(x):
        return x

    def J1(x):
        return 1

    U1 = 5

    def f2(x):
        return x**2

    def J2(x):
        return 2 * x

    U2 = 5

    # dimension 2*2
    def f3(x):
        return np.array([x[0] + x[1], x[0] - x[1]])

    def J3(x):
        return np.array([[1 + x[1], 1 - x[1]], [x[0] + 1, x[0] - 1]])

    U3 = np.array([5, 5])

    # dimension 2*1
    def f4(x):
        return np.array([x[0] + x[1]])

    def J4(x):
        return np.array([[1 + x[1], x[0] + 1]])

    U4 = np.array([5, 3])



####################################################################################

    c = 0
    for f, J, U0, in [[f1, J1, U1], [f2, J2, U2], [f3, J3, U3], [f4, J4, U4]]:
        c += 1
        r = Newton_Raphson(f, J, U0, 0.0001, N=1000)
        if not (abs(f(r)) < 0.005).all():
            print("Error with f{} : f(result) gives {}".format(c, f(r)))
            return False
    print("test_Newton_Raphson : OK")