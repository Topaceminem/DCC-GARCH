import numpy as np

def Q_average(tr):
    # return average of outer product of [eT,...e0]
    # et = [r(1t)/s(1t),...r(nt)/s(nt)]
    T = tr.shape[1]
    n = tr.shape[0]
    sum = np.zeros([n,n])
    for i in range(T):
        sum += np.outer(tr[:,i],tr[:,i])
    return sum/T

def Q_gen(tr,ab):
    # generate [QT,...Q0]
    Q_int = Q_average(tr)
    Q_list = [Q_int]
    T = tr.shape[1] - 1
    a = ab[0]
    b = ab[1]
    for i in range(T):
        et_1 = tr[:,T-i]
        Qt_1 = Q_list[0]
        Qt = (1.0-a-b)*Q_int + a*np.outer(et_1,et_1) + b*Qt_1
        Q_list = [Qt] + Q_list
    return Q_list

def R_gen(tr,ab):
    Q = Q_gen(tr,ab)
    # output [RT,...R0]
    R_list = []
    n = Q[0].shape[0]
    for i in Q:
        temp = 1.0/np.sqrt(np.abs(i))
        temp = temp * np.eye(n)
        R = np.dot(np.dot(temp,i),temp)
        R_list = R_list + [R]
    return R_list

def dcc_loss(tr, ab):
    R = R_gen(tr,ab)

    def dcc_loss_helper(tr=tr,R=R):
        loss = 0.0
        for i in range(len(R)):
            Ri = R[i]
            Ri_ = np.linalg.inv(Ri)
            ei = tr[:,i]
            loss += np.log(np.linalg.det(Ri)) + np.dot(np.dot(ei,Ri_),ei)
        # print('training loss %f' % loss)
        return loss

    return dcc_loss_helper()


def dcc_loss_gen():
    def loss1(tr):
        def loss(ab):
            return dcc_loss(tr, ab)
        return loss
    return loss1