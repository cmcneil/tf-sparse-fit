import unittest
import tf_sparse_fit as tfit
import numpy as np

class TestGetTrainingData(unittest.TestCase):
    def setUp(self):
        """
        Create a fake dataset. We set the numpy random seed so that this
        test isn't probabilistic.
        """
        self.n_features = 3207

        np.random.seed(0)
        self.X = np.random.randn(2579, self.n_features)
        self.X_test = np.random.randn(258, self.n_features)

        # The true function is a simple weight=1 sum of all the square numbered
        # features.]
        noise = np.random.randn(2579, 1)
        self.relevant_features = np.arange(50)**2
        self.Y = np.array([np.sum(self.X[:, self.relevant_features], axis=1)]).T
        print np.shape(self.Y)
        self.Y += noise
        self.Y_test = np.array([
                np.sum(self.X_test[:, self.relevant_features], axis=1)]).T

        self.true_W = np.zeros((self.n_features, 1))
        self.true_W[self.relevant_features] = 1

    def tearDown(self):
        """
        """
        pass

    def test_fit_tgd(self):
        """
        Test the threshold gradient descent fit. Show that relaxing the
        regularization parameter makes the fit worse.
        """
        fit = tfit.threshold_gradient_descent_fit(self.X, self.Y, self.X_test,
                                                  self.Y_test,
                                                  learning_rate=0.5,
                                                  threshold_grad_desc=0.9,
                                                  use_active_set=True,
                                                  max_iterations=2000,
                                                  verbose=False)
        fit_W = fit['weights'][0]
        self.assertTrue((np.abs(fit_W - self.true_W) < 0.15).all())

        ## For weaker regularization, the fit is not as good.
        fit = tfit.threshold_gradient_descent_fit(self.X, self.Y, self.X_test,
                                                  self.Y_test,
                                                  learning_rate=0.5,
                                                  threshold_grad_desc=0.2,
                                                  use_active_set=True,
                                                  max_iterations=2000,
                                                  verbose=False)
        fit_W = fit['weights'][0]
        self.assertFalse((np.abs(fit_W - self.true_W) < 0.3).all())

    def test_fit_group_lasso(self):
        """
        Test the group LASSO fit.
        """
        l1_params = 10.*np.ones(self.n_features)
        fit = tfit.group_lasso_fit(self.X, self.Y, self.X_test,
                                   self.Y_test,
                                   learning_rate=0.0001,
                                   max_iterations=500,
                                   l1_params=l1_params,
                                   verbose=False)
        fit_W = fit['weights'][0]
        print fit_W
        print self.true_W
        print np.abs(fit_W - self.true_W)
        print fit_W[self.relevant_features]
        self.assertTrue((np.abs(fit_W - self.true_W) < 0.4).all())




if __name__ == '__main__':
    unittest.main()
