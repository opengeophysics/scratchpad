import unittest
import numpy as np
from plugin import utils
from discretize import Tests


class TestSensitivity(unittest.TestCase):

    n = 100
    A = np.random.rand(int(n/2.), n)

    def function(self, x):
        return self.A.dot(x**2)**2

    def derivative_test(self, stencil):
        m = np.random.randn(self.n)
        sensitivity = utils.SensitivityFiniteDifference(
            function=lambda x: self.function(x), stencil=stencil
        )
        J = sensitivity(m)

        def function_and_deriv(x):
            return self.function(x), lambda v: J.dot(v)

        Tests.checkDerivative(
            function_and_deriv, x0=m, num=4, expectedOrder=2, plotIt=False
        )

    def test_centered_diff(self):
        self.derivative_test('centered')

    def test_forward_diff(self):
        self.derivative_test('forward')

    def test_backward_diff(self):
        self.derivative_test('backward')


if __name__ == '__main__':
    unittest.main()


