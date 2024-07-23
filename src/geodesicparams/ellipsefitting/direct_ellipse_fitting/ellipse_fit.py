from mpmath import matrix, eig, norm
from ..matrix_functions import element_pow, element_mul

def direct_ellipse_fit(data):
    x = data[0, :].T
    y = data[1, :].T
    
    d1 = matrix(x.rows, 3)
    d2 = matrix(x.rows, 3)
    
    d1[:, 0] = element_pow(x, 2)
    d1[:, 1] = element_mul(x, y)
    d1[:, 2] = element_pow(y, 2)

    d2[:, 0] = x
    d2[:, 1] = y
    d2[:, 2] = 1

    s1 = d1.T * d1
    s2 = d1.T * d2
    s3 = d2.T * d2
    t = - s3**(-1) * s2.T
    m = s1 + s2 * t

    c1 = matrix([[0, 0, 2], [0, -1, 0], [2, 0, 0]])
    m = c1**(-1) * m

    evalue, evec = eig(m)
    cond = 4 * element_mul(evec[0, :], evec[2, :]) - element_pow(evec[1, :], 2)
    for i in range(cond.cols):
        if cond[0, i] > 0:
            pos = i
            break

    a1 = evec[:, pos]
    a = matrix(2, 3)
    a[0, :] = a1.T
    a[1, :] = (t * a1).T

    a = a / norm(a)

    return a
