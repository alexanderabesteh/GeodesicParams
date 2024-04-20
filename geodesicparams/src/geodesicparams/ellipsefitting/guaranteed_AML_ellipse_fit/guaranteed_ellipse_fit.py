from mpmath import matrix, fabs, norm, eye, ones, zeros, sqrt
from .levenberg_marquardt import levenbergMarquardtStep 
from .line_search_step import lineSearchStep

class EllipseData:
  def __init__(self, use_pseudoinverse, theta_updated, lbd, k, damping_multiplier, gamma, numberOfPoints, f, I, alpha, data_points, tolDelta, tolCost, tolTheta,
               cost, t, delta, r, jacobian_matrix, jacobian_matrix_barrier, jacobian_matrix_full, hessian):
    self.use_pseudoinverse = use_pseudoinverse
    self.theta_updated = theta_updated
    self.lbd = lbd
    self.k = k
    self.damping_multiplier = damping_multiplier
    self.gamma = gamma
    self.numberOfPoints = numberOfPoints
    self.f = f
    self.I = I
    self.alpha = alpha
    self.data_points = data_points
    self.tolDelta = tolDelta
    self.tolCost = tolCost
    self.tolTheta = tolTheta
    self.cost = cost
    self.t = t
    self.delta = delta
    self.r = r
    self.jacobian_matrix = jacobian_matrix
    self.jacobian_matrix_barrier = jacobian_matrix_barrier
    self.jacobian_matrix_full = jacobian_matrix_full
    self.hessian = hessian

def guaranteedEllipseFit(t, data_points):
    keep_going = True
    fprim = matrix([[0, 0, 2], [0, -1, 0], [2, 0, 0]])
    maxIter = 200
    elp_data = EllipseData(False, False, 0.01, 0, 1.2, 0.00005, data_points.cols, zeros(6, 6), eye(6), 1e-3, 
                           data_points[0:2, :], 1e-7, 1e-7, 1e-7, zeros(1, maxIter), zeros(6, maxIter), zeros(6, maxIter), zeros(data_points.cols + 1, 1), zeros(data_points.cols, 6),
                         zeros(1, 6), matrix(data_points.cols + 1, 6), 0)
    elp_data.f[0:3, 0:3] = fprim

    t = t / norm(t)
    elp_data.t[0:3, elp_data.k] = t[0, :].T
    elp_data.t[3:6, elp_data.k] = t[1, :].T
    elp_data.delta[:, elp_data.k] = ones(6, 1)

    while keep_going and elp_data.k < maxIter:
        # allocate space for residuals (+1 to store barrier term) 
        elp_data.r = zeros(elp_data.numberOfPoints + 1, 1)
        # allocate space for the jacobian matrix based on AML component
        elp_data.jacobian_matrix = zeros(elp_data.numberOfPoints, 6)
        # allocate space for the jacobian matrix based on Barrier component
        elp_data.jacobian_matrix_barrier = zeros(1, 6)
        # grab the current parameter estimates
        t = elp_data.t[:, elp_data.k]
        # residuals computed on data points

        for i in range(elp_data.numberOfPoints):
            m = data_points[:, i]
            ux_j = matrix([[m[0]**2, m[0] * m[1], m[1]**2, m[0], m[1], 1]]).T
            dux_j = matrix([[2 * m[0], m[1], 0, 1, 0, 0], [0, m[0], 2 * m[1], 0, 1, 0]]).T

            a = ux_j * ux_j.T
            b = dux_j * dux_j.T
           
            tBt = (t.T * b * t)[0, 0]
            tAt = (t.T * a * t)[0, 0]
            elp_data.r[i] = sqrt(tAt / tBt)

            M = (a / tBt)
            xbits = b * (tAt / tBt**2)
            x = M - xbits
           
            # gradient for AML cost function 
            grad = x*t / sqrt(tAt/tBt)
            # build up jacobian matrix
            elp_data.jacobian_matrix[i, :] = grad.T
       
        tIt = (t.T * elp_data.I * t)[0]
        tFt = (t.T * elp_data.f * t)[0]

        elp_data.r[elp_data.numberOfPoints] = elp_data.alpha * (tIt / tFt)
        N = elp_data.I / tFt
        ybits = elp_data.f * (tIt) / (tFt)**2
        y = N - ybits
        # gradient for AML cost function 
        grad_penalty = 2 * elp_data.alpha * y * t
        # build up jacobian matrix
        elp_data.jacobian_matrix_barrier[0, :] = grad_penalty.T
       
        elp_data.jacobian_matrix_full[0:elp_data.numberOfPoints, :] = elp_data.jacobian_matrix 
        elp_data.jacobian_matrix_full[elp_data.numberOfPoints, :] = elp_data.jacobian_matrix_barrier

        elp_data.hessian = elp_data.jacobian_matrix_full.T * elp_data.jacobian_matrix_full
        elp_data.cost[elp_data.k] = (elp_data.r.T * elp_data.r)[0]

        if not elp_data.use_pseudoinverse:
            elp_data = levenbergMarquardtStep(elp_data)
        else:
            elp_data = lineSearchStep(elp_data)
       
        if (elp_data.t[:, elp_data.k + 1].T * elp_data.f * elp_data.t[:, elp_data.k + 1])[0, 0] <= 0:
            # from now onwards we will only use lineSearchStep to ensure
            # that we do not overshoot the barrier 
            elp_data.use_pseudoinverse = True
            elp_data.lbd = 0
            elp_data.t[:, elp_data.k + 1] = elp_data.t[:, elp_data.k]

            if (elp_data.k > 0):
                elp_data.t[:, elp_data.k] = elp_data.t[:, elp_data.k - 1]

        elif (min(norm(elp_data.t[:, elp_data.k + 1] - elp_data.t[:, elp_data.k]), norm(elp_data.t[:, elp_data.k + 1] + elp_data.t[:, elp_data.k])) 
                < elp_data.tolTheta and elp_data.theta_updated):
            keep_going = False
        elif (fabs(elp_data.cost[elp_data.k] - elp_data.cost[elp_data.k + 1]) < elp_data.tolCost and elp_data.theta_updated):
            keep_going = False
        elif (norm(elp_data.delta[:, elp_data.k + 1]) < elp_data.tolDelta and elp_data.theta_updated):
            keep_going = False

        elp_data.k = elp_data.k + 1

    theta = elp_data.t[:, elp_data.k]
    theta = theta / norm(theta)

    return theta

