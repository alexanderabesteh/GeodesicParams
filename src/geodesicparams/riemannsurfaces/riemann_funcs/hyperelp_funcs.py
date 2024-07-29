from mpmath import exp, sin, cos, pi, re, im, mp

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

def naive_theta_genus2(z, tau, precision = 53):
    mp.prec = precision  # Set precision
   
    # Assume 2 Im(tau_3) <= Im(tau_1) <= Im(tau_2) (Minkowski-reduced)
    RR = mp
    # Compute B
    B = 2 * precision * RR.log(10) / RR.pi + 3
    # Get the precision right to counter rounding error
    # In genus 1 this was precision + 7*log(B), let's do precision + 20*log(B), just in case
    CC = mp

    q1 = mp.exp(CC.j * mp.pi * tau[0, 0])
    q1sq = q1 ** 2
    q2 = mp.exp(CC.j * mp.pi * tau[1, 1])
    q2sq = q2 ** 2
    q3 = mp.exp(CC.j * mp.pi * tau[0, 1])
    q3sq = q3 ** 2

    w1 = mp.exp(CC.j * mp.pi * z[0])
    w1sq = w1 ** 2
    w2 = mp.exp(CC.j * mp.pi * z[1])
    w2sq = w2 ** 2

    # 4 theta-constants with a=0
    result = CC.mpc(1)

    # n=0
    q1m2 = q1
    q12mminus2 = q1sq  # At the beginning of the loop, it contains 2(m+1)-2
    r1 = w1sq + 1 / w1sq
    r1m = q1 * r1
    r1mminus1 = 2
    when_to_stop = int(mp.ceil(mp.sqrt(B / mp.im(tau[0, 0]))) + 3)

    for m in range(1, when_to_stop + 1):
        result += r1m
        q12mminus2timesq1 = q12mminus2 * q1  # Mini-optimization
        bubu = r1m
        r1m = r1 * r1m * q12mminus2timesq1 - r1mminus1 * (q12mminus2 ** 2)
        r1mminus1 = bubu
        # Theta constants
        term = 2 * q1m2
        q1m2 *= q12mminus2timesq1
        q12mminus2 *= q1sq

    # m,n >=1
    v1 = q3sq + 1 / q3sq
    vnminus1 = 2
    vn = v1
    q2to2nminus2 = 1
    q2n2 = q2
    q3to2n = q3sq
    s1 = w2sq + 1 / w2sq
    w1w2sq = w1sq * w2sq
    w1invw2sq = w2sq / w1sq
    q2s1 = q2 * s1
    q1r1 = q1 * r1
    q1q2 = q1 * q2  # Small optimisations
    betas = [[q1r1, q1q2 * q3sq * (w1w2sq + 1 / w1w2sq)], [CC.mpc(2), q2s1]]
    betaprimes = [[q1r1, (q1q2 / q3sq) * (w1invw2sq + 1 / w1invw2sq)], [CC.mpc(2), q2s1]]

    for n in range(1, int(mp.ceil(mp.sqrt(B / mp.im(tau[1, 1])))) + 4):
        if n > 3:
            when_to_stop = B - (n - 3) ** 2 * mp.im(tau[1, 1])
        else:
            when_to_stop = B
        if when_to_stop <= 0:
            when_to_stop = 0
        when_to_stop = mp.ceil(mp.sqrt(when_to_stop / mp.im(tau[0, 0]))) + 3

        q1to2mminus2 = q1sq  # This squared gives q^(4m-4)
        alphamminus1 = 2 * q2n2
        alpham = q1 * q2n2 * vn

        term = betas[1][1]  # Not betas+betaprimes (since m=0 we only add the term once)
        result += term

        alphazm = betas[0][1]
        alphaprimezm = betaprimes[0][1]
        alphazmminus1 = betas[1][1]
        alphaprimezmminus1 = betaprimes[1][1]

        for m in range(1, int(when_to_stop) + 1):
            term = 2 * alpham
            bubu = alpham
            alpham = vn * (q1to2mminus2 * q1) * alpham - (q1to2mminus2 ** 2) * alphamminus1
            alphamminus1 = bubu

            term = alphazm + alphaprimezm
            result += term

            r1Xq1to2mminus2Xq1 = r1 * q1to2mminus2 * q1
            bubu = alphazm
            alphazm = alphazm * r1Xq1to2mminus2Xq1 * q3to2n - (q1to2mminus2 * q3to2n) ** 2 * alphazmminus1
            alphazmminus1 = bubu
            bubu = alphaprimezm
            alphaprimezm = alphaprimezm * r1Xq1to2mminus2Xq1 / q3to2n - (q1to2mminus2 / q3to2n) ** 2 * alphaprimezmminus1
            alphaprimezmminus1 = bubu

            q1to2mminus2 *= q1sq

        bubu = vn
        vn = vn * v1 - vnminus1
        vnminus1 = bubu
        q2to2nminus2 *= q2sq
        q2n2 *= q2to2nminus2 * q2

        s1Xq2to2nminus2Xq2 = s1 * q2to2nminus2 * q2
        bubu = [betas[0][1], betas[1][1]]
        betas[0][1] = betas[0][1] * s1Xq2to2nminus2Xq2 * q3sq - betas[0][0] * (q2to2nminus2 * q3sq) ** 2
        betas[1][1] = betas[1][1] * s1Xq2to2nminus2Xq2 - q2to2nminus2 ** 2 * betas[1][0]
        betas[0][0] = bubu[0]
        betas[1][0] = bubu[1]
        bubu = [betaprimes[0][1], betaprimes[1][1]]
        betaprimes[0][1] = betaprimes[0][1] * s1Xq2to2nminus2Xq2 / q3sq - betaprimes[0][0] * (q2to2nminus2 / q3sq) ** 2
        betaprimes[1][1] = betaprimes[1][1] * s1Xq2to2nminus2Xq2 - q2to2nminus2 ** 2 * betaprimes[1][0]
        betaprimes[0][0] = bubu[0]
        betaprimes[1][0] = bubu[1]
        q3to2n *= q3sq

    result = mp.mpc(result)
    return result

def theta_char(z, tau, char, precision = 53):
    mp.prec = precision

    z = mp.matrix(z)
    tau = mp.matrix(tau)
    g = mp.matrix(char[0]); h = mp.matrix(char[1])
    exp_factor = mp.exp((1j * mp.pi * g.T * (tau * g + 2 * z + 2 * h))[0])
    result = exp_factor * naive_theta_genus2(z + tau * g + h, tau, precision)

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
