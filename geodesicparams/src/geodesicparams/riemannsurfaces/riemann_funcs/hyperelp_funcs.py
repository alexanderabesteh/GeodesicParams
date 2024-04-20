from mpmath import exp, sin, cos, pi, re, im

def hyp_theta(z, riemannM, minMax = 5):

    g = [1/2, 1/2]; h = [0, 1/2]
    result = 0

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j])# + 2 * z[i] + 2 * h[i] 
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            result += exp(1j * pi * char_sum)

    return result

def hyp_theta_RR(xR, xI, wR, wI, l, riemannM, minMax = 5):

    g = [1/2, 1/2]
    h = [0, 1/2]
    result = 0
    varR = [xR, wR]
    varI = [xI, wI]
    riemannR = riemannM.apply(re)
    riemannI = riemannM.apply(im)

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sumExp = 0
            char_sumSin = 0

            for i in range(2):
                tau_sumI = 0
                tau_sumR = 0

                for j in range(2):
                    tau_sumI += -riemannI[i, j] * (m[j] + g[j])
                    tau_sumR += riemannR[i, j] * (m[j] + g[j])

                char_sumExp += (m[i] + g[i]) * (tau_sumI - 2 * varI[i])
                char_sumSin += (m[i] + g[i]) * (tau_sumR + 2 * varR[i] + 2 * h[i])

            result -= (exp(pi * char_sumExp) * sin(pi * char_sumSin) * 2 * pi * (m[l] + g[l]))
    return result
 
def hyp_theta_IR(xR, xI, wR, wI, l, riemannM, minMax = 5):

    g = [1/2, 1/2]; h = [0, 1/2]
    result = 0
    varR = [xR, wR]
    varI = [xI, wI]
    riemannR = riemannM.apply(re)
    riemannI = riemannM.apply(im)

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sumExp = 0
            char_sumCos = 0

            for i in range(2):
                tau_sumI = 0
                tau_sumR = 0

                for j in range(2):
                    tau_sumI += -riemannI[i, j] * (m[j] + g[j])
                    tau_sumR += riemannR[i, j] * (m[j] + g[j])

                char_sumExp += (m[i] + g[i]) * (tau_sumI - 2 * varI[i])
                char_sumCos += (m[i] + g[i]) * (tau_sumR + 2 * varR[i] + 2 * h[i])
            
            result += exp(pi * char_sumExp) * cos(pi * char_sumCos) * 2 * pi * (m[l] + g[l])
    return result

def sigma1(z, riemannM, minMax = 5):

    g = [1/2, 1/2]
    h = [0, 1/2]
    result = 0

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j]) 
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            result += exp(1j * pi * char_sum) * 2 * pi * 1j * (m1 + g[0])

    return result

def sigma2(z, riemannM, minMax = 5):

    g = [1/2, 1/2]
    h = [0, 1/2]
    result = 0

    for m1 in range(-minMax, minMax + 1):
        for m2 in range(-minMax, minMax + 1):
            m = [m1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j]) 
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])
            result += exp(1j * pi * char_sum) * 2 * pi * 1j * (m2 + g[1])

    return result
