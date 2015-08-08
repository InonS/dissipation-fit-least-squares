'''
@author: Inon Sharony
@contact: www.tau.ac.il/~inonshar
@organization: Tel Aviv University Department of Chemical Physics
@since:  Aug 5, 2015
'''
from numpy import exp
from random import gauss
import unittest

from src.FitDissipationOrdinaryLeastSquares import FitDissipationOrdinaryLeastSquares


class FitDissipationOrdinaryLeastSquaresTest(unittest.TestCase):
    """
    @see: Web resources:
        https://docs.python.org/3/library/unittest.html#classes-and-functions
        http://pymotw.com/2/unittest/
        http://docs.python-guide.org/en/latest/writing/tests/
        http://www.openp2p.com/pub/a/python/2004/12/02/tdd_pyunit.html
        http://www.drdobbs.com/testing/unit-testing-with-python/240165163
        http://www.diveintopython3.net/unit-testing.html
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testExponentialDecay(self):
        A = 1.2
        g = 5 / 7
        N = 750
        SNR = 75
        logAbsErrTol = 2
        fdols = FitDissipationOrdinaryLeastSquares()
        fdols.__init__(x=[5 * (t / N) for t in range(N)],
                       y=[A * exp(-g * 5 * (t / N)) + gauss(0, A / SNR) for t in range(N)], SNR=SNR)
        optimizeResult, R2, Chi2 = fdols.run()
        self.assertTrue(
            optimizeResult.success, msg="Returned success value is not True")
        self.assertGreater(
            optimizeResult.x[0], 0, msg="Amplitude is not positive")
        self.assertGreater(
            optimizeResult.x[1], 0, msg="Dissipation coefficient is not positive")
        self.assertAlmostEqual(
            optimizeResult.x[0], A, places=logAbsErrTol, msg="Amplitude is not almost equal to generating value")
        self.assertAlmostEqual(
            optimizeResult.x[1], g, places=logAbsErrTol, msg="Dissipation coefficient is not almost equal to generating value")
        self.assertGreater(
            R2, 0.97, msg="Coefficient of determination is not above 97%")
        self.assertGreater(
            Chi2, 1 / 3, msg="Goodness of fit is not above 33.33%")
        self.assertLess(Chi2, 3, msg="Goodness of fit is not below 300%")

    @classmethod
    def getSubTestSet(cls):
        subTests = {''}
        for m in cls.__dict__:
            if not callable(m):
                break
            if m.__name__ == "setUp":
                break
            if m.__name__ == "tearDown":
                break
            subTests.append(m)
        return subTests

if __name__ == "__main__":
    from sys import argv
    argv = argv if len(argv) <= 1\
        else FitDissipationOrdinaryLeastSquaresTest.getSubTestSet(FitDissipationOrdinaryLeastSquaresTest)
    unittest.main()
