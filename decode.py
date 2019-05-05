import numpy as np
from scipy import optimize
from scipy import sparse
import math


def decode(A, y):
    """
    min [ 0 1 ]^T[ g  t ]

    s.t.
    [ -A -I ][ g ] <= [ -y ]
    [  A -I ][ t ]    [  y ]
    """

    A_ub = np.concatenate((
        np.concatenate((A * -1, A), axis=0),
        np.concatenate((
            -1 * np.identity(A.shape[0]),
            -1 * np.identity(A.shape[0])), axis=0)), axis=1)

    b_ub = np.concatenate((-1 * y, y), axis=0)
    lp_coeff = np.concatenate((np.zeros(A.shape[1]), np.ones(A.shape[0])))

    return optimize.linprog(
        lp_coeff, A_ub=A_ub, b_ub=b_ub,
        bounds=[(None, None) for _ in range(A.shape[0] + A.shape[1])])


def form(s=256, e=0.1, mul=2):
    """
    """

    A = np.random.normal(size=(mul * s, s), scale=math.sqrt(1 / s))
    x = np.random.normal(size=(s, 1))
    e = sparse.random(mul * s, 1, density=e).A
    y = np.matmul(A, x)  # + e

    return A, x, e, y


def test(S, e, mul, task):

    task.start("Testing n={}, m={}n".format(S, mul))

    A, x, e, y = form(s=S, e=e, mul=mul)

    opt = decode(A, y)

    """
    print("e:")
    print(e)

    print("original:")
    print(x[:, 0])
    print("obj:")
    print(opt.fun)
    print(opt.x[:S])
    print('err:')
    """

    err = np.linalg.norm(x[:, 0] - opt.x[:S]) / np.linalg.norm(x[:, 0])

    task.done()

    return err


if __name__ == '__main__':

    import syllabus

    n = 100
    mscale = 2

    main = syllabus.BasicTaskApp().start(
        "LP Decoding: m={}, mscale={}".format(m, mscale))
    main.add_task(25)
