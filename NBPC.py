import numpy as np
import math

class NBPC():

    #standard normal distribution
    def get_propability_gaussian(self,feature_value,feature_num):
        mean = self.mean_variance[feature_num][0]
        variance = self.mean_variance[feature_num][1]
        a = 1/(math.sqrt(2*math.pi*(variance*variance)))
        b = -1*((feature_value-mean)*(feature_value-mean))
        c = 2*(variance*variance)
        propability = a*math.exp(b/c)
        return propability

    def get_propability(self,sample):
        prop = self.get_propability_gaussian(sample[0], 0)
        for i in range(1, len(sample)):
            prop *= self.get_propability_gaussian(sample[i], i)
        return prop


    def calculate_t(self,X):
        propability_values = list()
        for x in X:
            prop = self.get_propability_gaussian(x[0],0)
            for i in range(1,len(x)):
                prop*=self.get_propability_gaussian(x[i],i)
            propability_values.append(prop)
        self.t = min(propability_values)

    def __init__(self):
        self.t = None

    def fit(self, X, y=None):
        """"
        :param X:  array-like, shape = [n_samples, n_features]
        :param y: array, shape = [n_samples] target values for X
        :return:
        """
        n_samples, n_features = X.shape
        #mean variance matrix for every feature.
        # [n,0] -> mean value for feature n
        # [n,1] -> variance for feature n
        self.mean_variance = np.zeros([n_features,2])
        for i in range(n_features):
            self.mean_variance[i][0] = np.mean(X[:, i])
            #add 1*10^-10 noise if variance is zero
            self.mean_variance[i][1] = np.var(X[:, i]) if np.var(X[:, i]) != 0 else 0.0000000001
        self.calculate_t(X)

    def predict(self,X):
        predicitions = list()
        for sample in X:
            if self.get_propability(sample) >= self.t:
                predicitions.append(1)
            else:
                predicitions.append(-1)
        return np.array(predicitions)
