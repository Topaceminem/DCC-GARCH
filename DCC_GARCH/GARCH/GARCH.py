import numpy as np
from scipy.optimize import minimize
from DCC_GARCH.GARCH.GARCH_loss import garch_process


class GARCH():

    def __init__(self, p=1, q=1, max_itr=3, early_stopping=True):
        # p the lag of r_t, q the lag of s_t
        self.p = p
        self.q = q
        theta0 = [0.005] + [0.1 for i in range(p)] + [0.1 for i in range(p)] + [0.85 for i in range(q)]
        self.theta = np.array(theta0)
        self.max_itr = max_itr
        self.early_stopping = early_stopping
        def ub(x):
            return 1. - x[1] - 0.5*x[2] - x[3]
        def lb1(x):
            return x[1] + x[2]
        def lb2(x):
            return x[0]
        def lb3(x):
            return x[1]
        def lb4(x):
            return x[3]
        self.constraints = [{'type':'ineq', 'fun':ub},{'type':'ineq', 'fun':lb1},
                            {'type':'ineq', 'fun':lb2},{'type':'ineq', 'fun':lb3},
                            {'type':'ineq', 'fun':lb4}]
        self.method = 'COBYLA'

    def set_theta(self, theta):
        self.theta = np.array(theta)

    def get_theta(self):
        return self.theta

    def get_p(self):
        return self.p

    def get_q(self):
        return self.q

    def set_loss(self, loss_func):
        "loss function L is a meta-function, s.t. L(r) = f(theta)."
        self.loss_func = loss_func

    def get_loss_func(self):
        if self.loss_func is None:
            raise Exception("No Loss Function Found!")
        else:
            return self.loss_func

    def set_max_itr(self, max_itr):
        self.max_itr = max_itr

    def get_max_itr(self):
        return self.max_itr

    def set_method(self, method):
        self.method = method

    def get_method(self):
        return self.method

    def fit(self, train_data):  # train_data: [rT,...r0]
        tr = train_data

        # Optimize using scipy and save theta
        tr_losses = []
        j = 0
        count = 0
        while j < self.get_max_itr():
            j += 1
            theta0 = self.get_theta()
            res = minimize(self.get_loss_func()(tr), theta0, method=self.method,
                           options={'disp': True}, constraints=self.constraints)
            theta = res.x
            self.set_theta(theta)

            tr_loss = self.get_loss_func()(tr)(theta)
            tr_losses.append(tr_loss)
            print("Iteration: %d. Training loss: %.3E." % (j, tr_loss))

            # Early stopping
            if self.early_stopping is True:
                if j > 10:
                    if abs(tr_losses[-1] - tr_losses[-2]) / tr_losses[-2] < 0.0001:
                        count += 1
                        if count >= 2:
                            print("Early Stopping...")
                            return tr_losses
        return tr_losses

    def sigma(self, y):  # test data: [rT,...r0]
        s = garch_process(y, self.theta, self.p, self.q)
        return np.array(s)