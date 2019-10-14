from lib import *
import numpy as np

def garch_process(r, theta, p=1, q=1):
    w, alpha, gamma, beta = theta[0], theta[1:1 + p], theta[1 + p:1 + p + p], theta[1 + p + p:]
    if len(gamma) is not q:
        raise Exception('Parameter Length Incorrect!')
    r = np.array(r)
    T = len(r) - 1

    def garch_update(s, r, t, alpha, beta, gamma, p=p, q=q, T=T):
        "s = [st-1,...s0], r = [rT,...,r0], t is time" \
        "alpha, beta and gamma are from above" \
        "returns new_s = [st,...,s0]"
        r_temp = r[T - t + 1:T - t + 1 + q]  # [rt-1,...,rt-q]
        s_temp = s[0:p]  # [st-1,...st-p]

        var = np.array(s_temp) ** 2
        r_squared = np.array(r_temp) ** 2
        gjr = r_squared*(np.array(r_temp)<0)
        st = np.sqrt(np.abs(np.dot(np.array(beta), var) + np.dot(np.array(alpha), r_squared)
                      + np.dot(np.array(gamma), gjr) + w))

        new_s = [st] + s

        return new_s #[sT,...,s0]

    #"Initialize values of s and m as data variance and mean"
    s_int = np.std(r)
    L = max(p, q)
    s = [s_int for i in range(0, L)]

    for t in range(L, T + 1):
        s = garch_update(s, r, t, alpha, beta, gamma)

    return s


def garch_loss(r, theta, p, q):
    s = garch_process(r, theta, p, q)

    def garch_loss_helper(r=r, s=s):
        s = np.array(s)
        loss = 0.0
        for i in range(len(r)):
            loss += np.log(s[i] ** 2) + (r[i]/s[i])**2

    #print('training loss %f' % loss)
        return loss

    return garch_loss_helper()


def garch_loss_gen(p=1, q=1):
    def loss1(r):
        def loss(theta):
            return garch_loss(r, theta, p, q)
        return loss
    return loss1
