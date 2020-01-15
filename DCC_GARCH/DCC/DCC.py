import numpy as np
from scipy.optimize import minimize
from DCC_GARCH.DCC.DCC_loss import Q_gen, Q_average

class DCC():

    def __init__(self, max_itr=2, early_stopping=True):
        self.max_itr = max_itr
        self.early_stopping = early_stopping
        self.ab = np.array([0.5, 0.5])
        self.method =  'SLSQP'
        def ub(x):
            return 1. - x[0] - x[1]
        def lb1(x):
            return x[0]
        def lb2(x):
            return x[1]
        self.constraints = [{'type':'ineq', 'fun':ub},{'type':'ineq', 'fun':lb1},{'type':'ineq', 'fun':lb2}]

    def set_ab(self,ab): # ndarray
        self.ab = ab

    def get_ab(self):
        return self.ab

    def set_method(self,method):
        self.method = method

    def set_loss(self, loss_func):
        #"loss function L is a meta-function, s.t. L(r) = f(theta)."
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

    def fit(self, train_data):
        #train_data: numpy.array([[e1_T,...e1_0],\
        #                         [e2_T,...e2_0],\
        #                         ...,
        #                         [en_T,...en_0]])

        tr = train_data

        # Optimize using scipy and save theta
        tr_losses = []
        j = 0
        count = 0
        while j < self.get_max_itr():
            j += 1
            ab0 = np.array(self.get_ab())
            res = minimize(self.get_loss_func()(tr), ab0, method=self.method,
                           options={'disp': True},constraints=self.constraints)
            ab = res.x
            self.set_ab(ab)

            tr_loss = self.get_loss_func()(tr)(ab)
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


    def Q(self,y):
        Q = Q_gen(y,self.ab)
        return Q

    def Q_bar(self,y):
        return Q_average(y)