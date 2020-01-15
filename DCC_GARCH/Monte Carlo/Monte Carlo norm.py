import numpy as np
# for GARCH(1,1) only
def s_MT(theta,rT,sT): #(ndarray, num, num)
    w, alpha, gamma, beta = theta[0], theta[1], theta[2], theta[3]
    return np.sqrt(w + alpha*(rT**2) + gamma*(rT**2)*(rT<0) + beta*(sT**2))

def Q_MT(ab,Q_bar,eT,QT): #(ndarray, ndarray, ndarray, ndarray)
    a, b = ab[0], ab[1]
    return (1 - a - b) * Q_bar + a * np.outer(eT,eT) + b * QT

def R_MT(ab,Q_bar,eT,QT): #(ndarray, ndarray, ndarray, ndarray)
    Q = Q_MT(ab,Q_bar,eT,QT)
    n = Q.shape[0]
    temp = 1 / np.sqrt(np.abs(Q))
    temp = temp * np.eye(n)
    return np.dot(np.dot(temp, Q), temp)

def H_MT(theta_list,rT_list,sT_list,ab,Q_bar,eT,QT):
    # (list of ndarray, list, list, ndarray, ndarray, ndarray, ndarray)
    n = len(theta_list)
    S_forward = np.zeros([n,n])
    s_forward = []
    for i in range(n):
        s_forward_i = s_MT(theta_list[i],rT_list[i],sT_list[i])
        S_forward[i,i] = s_forward_i
        s_forward = s_forward + [s_forward_i]
    R_forward = R_MT(ab,Q_bar,eT,QT)
    return np.dot(np.dot(S_forward, R_forward), S_forward), np.array(s_forward)

def gen_new(theta_list,rT_list,sT_list,ab,Q_bar,eT,QT):
    H, s = H_MT(theta_list,rT_list,sT_list,ab,Q_bar,eT,QT)
    h = np.linalg.cholesky(H)
    n = h.shape[0]
    r = np.dot(h, np.random.randn(n))
    e = r / s
    Q = Q_MT(ab,Q_bar,eT,QT)

    return r, s, e, Q

def gen_new_r_squence(theta_list,rT_list,sT_list,ab,Q_bar,eT,QT,m):
    # output [rT+m,...,rT+1,rT]
    r_squen = [np.array(rT_list)]
    s_squen = [np.array(sT_list)]
    e_squen = [eT]
    Q_squen = [QT]
    for i in range(m):
        r_i, s_i, e_i, Q_i = gen_new(theta_list,r_squen[0],s_squen[0],ab,Q_bar,e_squen[0],Q_squen[0])
        r_squen = [r_i] + r_squen
        s_squen = [s_i] + s_squen
        e_squen = [e_i] + e_squen
        Q_squen = [Q_i] + Q_squen
    return r_squen