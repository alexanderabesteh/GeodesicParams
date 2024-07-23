from mpmath import matrix, norm, lu_solve

def levenbergMarquardtStep(struct):
    
    t = struct.t[:, struct.k]

    jacobian_matrix = struct.jacobian_matrix
    jacobian_matrix_barrier = struct.jacobian_matrix_barrier
    r = struct.r
    I = struct.I
    lbd = struct.lbd
    delta = struct.delta[:, struct.k]
    damping_multiplier = struct.damping_multiplier
    f = struct.f
    current_cost = struct.cost[struct.k]
    data_points = struct.data_points
    alpha = struct.alpha
    numOfPoints = struct.numberOfPoints
    
    temp = matrix(numOfPoints + 1, 6)
    temp[0:numOfPoints, :] = jacobian_matrix
    temp[numOfPoints, :] = jacobian_matrix_barrier
    jacob = temp.T * r
    
    update_a = lu_solve( - (struct.hessian + lbd * I), jacob)
    
    update_b = lu_solve( - (struct.hessian + (lbd / damping_multiplier) * I), jacob)

    t_potential_a = t + update_a
    t_potential_b = t + update_b
        
    cost_a = 0
    cost_b = 0
    for i in range(numOfPoints):
        m = data_points[:, i]
        ux_j = matrix([[m[0]**2, m[0] * m[1], m[1]**2, m[0], m[1], 1]]).T
        dux_j = matrix([[2 * m[0], m[1], 0, 1, 0, 0], [0, m[0], 2 * m[1], 0, 1, 0]]).T

        a = ux_j * ux_j.T
        b = dux_j * dux_j.T

        t_aBt_a = (t_potential_a.T * b * t_potential_a)[0, 0]
        t_aAt_a = (t_potential_a.T * a * t_potential_a)[0, 0]

        t_bBt_b = (t_potential_b.T * b * t_potential_b)[0, 0]
        t_bAt_b = (t_potential_b.T * a * t_potential_b)[0, 0]

        cost_a += t_aAt_a / t_aBt_a
        cost_b += t_bAt_b / t_bBt_b

    t_aIt_a = (t_potential_a.T * I * t_potential_a)[0, 0]
    t_aFt_a = (t_potential_a.T * f * t_potential_a)[0, 0]
    
    t_bIt_b = (t_potential_b.T * I * t_potential_b)[0, 0]
    t_bFt_b = (t_potential_b.T * f * t_potential_b)[0, 0]
    
    cost_a += (alpha * (t_aIt_a / t_aFt_a))**2;
    cost_b += (alpha * (t_bIt_b / t_bFt_b))**2;
    
   # print(cost_a)
   # print(cost_b)
   # print(current_cost)
    #print("")
    if (cost_a >= current_cost and cost_b >= current_cost):
        # neither update reduced the cost
        struct.theta_updated = False
        # no change in the cost
        struct.cost[struct.k + 1] = current_cost
        # no change in parameters
        struct.t[:, struct.k + 1] = t
        # no changes in step direction
        struct.delta[:, struct.k + 1] = delta
        # next iteration add more Identity matrix
        struct.lbd = lbd * damping_multiplier
    elif (cost_b < current_cost):
        # update 'b' reduced the cost function
        struct.theta_updated = True
        # store the new cost
        struct.cost[struct.k + 1] = cost_b
        # choose update 'b'
        struct.t[:, struct.k + 1] = t_potential_b / norm(t_potential_b)
        # store the step direction
        struct.delta[:, struct.k + 1] = update_b
        # next iteration add less Identity matrix
        struct.lbd = lbd / damping_multiplier
    else:
        # update 'a' reduced the cost function
        struct.theta_updated = True
        # store the new cost
        struct.cost[struct.k + 1] = cost_a
        # choose update 'a'
        struct.t[:, struct.k + 1] = t_potential_a / norm(t_potential_a)
        # store the step direction
        struct.delta[:, struct.k + 1] = update_a
        # keep the same damping for the next iteration
        struct.lbd = lbd

    return struct
