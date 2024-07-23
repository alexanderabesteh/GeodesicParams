from mpmath import matrix, norm
from numpy import array
from numpy.linalg import pinv 

def lineSearchStep(struct):

    t = struct.t[:, struct.k]

    jacobian_matrix = struct.jacobian_matrix
    jacobian_matrix_barrier = struct.jacobian_matrix_barrier
    r = struct.r
    I = struct.I
    delta = struct.delta[:, struct.k]
    tolDelta = struct.tolDelta
    f = struct.f
    current_cost = struct.cost[struct.k]
    data_points = struct.data_points
    alpha = struct.alpha
    gamma = struct.gamma
    numOfPoints = struct.numberOfPoints
    hessian = struct.hessian

    temp = matrix(numOfPoints + 1, 6)
    temp[0:numOfPoints, :] = jacobian_matrix
    temp[numOfPoints, :] = jacobian_matrix_barrier

    jacob = temp.T * r
    tFt = (t.T * f * t)[0]
   
    update = - matrix(pinv(array(hessian.tolist(), dtype = float), 1e-20)) * jacob

    frac = 0.5
    
    while True:
        t_potential = t + frac * update
        delta = frac * update
        frac /= 2
        cost = 0
        for i in range(numOfPoints):
            m = data_points[:, i]
            ux_j = matrix([[m[0]**2, m[0] * m[1], m[1]**2, m[0], m[1], 1]]).T
            dux_j = matrix([[2 * m[0], m[1], 0, 1, 0, 0], [0, m[0], 2 * m[1], 0, 1, 0]]).T

            a = ux_j * ux_j.T
            b = dux_j * dux_j.T

            tBt = (t_potential.T * b * t_potential)[0, 0]
            tAt = (t_potential.T * a * t_potential)[0, 0]
               
            cost +=  tAt / tBt

        tIt = (t_potential.T * I * t_potential)[0, 0]
        tFt = (t_potential.T * f * t_potential)[0, 0]
    
        cost += (alpha * (tIt / tFt))**2

        if (t_potential.T * f * t_potential > 0 and (cost < (1 - frac * gamma) * current_cost) or norm(delta) < tolDelta):
            break

    struct.theta_update = True
    struct.t[:, struct.k + 1] = t_potential / norm(t_potential)
    struct.delta[:, struct.k + 1] = delta
    struct.cost[struct.k + 1] = cost
 
    return struct
    
