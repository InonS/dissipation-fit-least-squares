'''
@author: Inon Sharony
@contact: www.tau.ac.il/~inonshar
@organization: Tel Aviv University Department of Chemical Physics
@since:  Aug 2, 2015
'''
from logging import basicConfig, info, debug
from math import exp
from numpy import linspace, array, square, var
from random import gauss

from matplotlib.pyplot import plot, show
from scipy.optimize import minimize


def exponentialDecay(t, Amp, gamma):
    return Amp * exp(-gamma * t)


class DissipationFitLeastSquares():
    """
    Computes effective dissipation coefficient
    for an exponentially decaying (scalar) signal
    with random Gaussian Markovian noise.

    model:
    f(t;Amp,gamma) = Amp * exp(-\gamma * t)

    noisy signal:
    y(t) = f(t) + eps

    noise properties:
    E[eps] = 0
    STDEV[eps] = Amp * relNoise    

    parameters:
    thetas = [ A, g ]

    target function:
    function(t;A,g) = Sum over t of [ ( y(t) - f(t;A,g) )^2 ]
                           = Sum over t of [ y^2 -2*y*f + f^2 ]

    target Jacobian (not necessary, but speeds convergence up):
    jacobian(t;A,g) = Sum over t of [ d(target_function(t;A,g)) / dA , d(target_function(t;A,g)) / dg ]

    target Hessian (not necessary, but speeds convergence up):
    (couldn't be bothered...)
    """

    def __init__(self, x=None, y=None, n=1000, Amp0=1.0, gamma0=10.0, relNoise=0.04):

        basicConfig(level="DEBUG")

        self.n = n
        self.x = x if x is not Null else linspace(0, 1, self.n)

        self._A_ = Amp0
        self._g_ = gamma0
        dof = self.n - len([self._A_, self._g_])

        self.y = y if y is not None else self.generateNoisySignal(relNoise)

        info("{}\nn = {}\nx = {}\ny = {}\nAmp0 = {}\ngamma0 = {}\nrelNoise = {}"
             .format(self, self.n, self.x.shape, self.y.shape, self._A_, self._g_, relNoise))

        self.varData = var(self.y)
        self.SStot = self.varData * self.n
        debug("(Var[y] = {:.5g}) = ({:s} = {:.5g}) / (dof = {:d})"
              .format(self.varData, r'${SS}_{tot}$', self.SStot, dof))

    def generateNoisySignal(self, relNoise):
        fx = [exponentialDecay(t, self._A_, self._g_) for t in self.x]
        rx = [(exponentialDecay(t, self._A_, self._g_) +
               gauss(0, self._A_ * relNoise)) for t in self.x]
        return array(rx)

    def function(self, thetas):
        s = 0
        for i in range(self.n):
            s += square(self.y[i])\
                - 2 * self.y[i] * thetas[0] * exp(-thetas[1] * self.x[i])\
                + square(thetas[0]) * exp(-2 * thetas[1] * self.x[i])
        return s

    def jacobian(self, thetas):
        s = 0
        for i in range(self.n):
            s += 2 * exp(-thetas[1] * self.x[i])\
                * (self.y[i] - thetas[0] * exp(-thetas[1] * self.x[i]))\
                * array([-1, self.x[i] * thetas[0]])
        return s

    def run(self, disp=True):

        optimizeResult = minimize(fun=self.function,
                                  x0=array([self._A_, self._g_]),
                                  method="SLSQP", jac=self.jacobian,
                                  bounds=[(0, None), (0, None)],
                                  options={'disp': disp})

        info(repr(optimizeResult))

        SSres = optimizeResult.get('fun')
        # $R^2 \rightarrow 1^{-}$ : perfect fit, $R^2 \rightarrow 0^{+}$ : no fit
        info("coefficient of determination ({:s}) = {:.2%}"
             .format(r'$R^2$', 1 - SSres / self.SStot))
        # chi^2 >> 1 : under-fitting, chi^2 << 1 : over-fitting
        info("goodness of fit ({:s}, {:s}{:d}) = {:.2%}".format(
            r'$\Chi^2_{\ni}$', r'$\ni=$', self.n, SSres / self.varData))

        A, g = optimizeResult.x
        plot(self.x, self.y, 'r.', self.x,
             [exponentialDecay(t, A, g) for t in self.x], 'b_')
        show()


def main(args):
    """
    python -m DissipationFitLeastSquares [x] [[y] [[[n=1000] [[[[Amp0=1.0] [[[[[gamma0=10.0] [[[[[[relNoise=0.04]]]]]]]]]]]]]]]]
    """
    dfls = DissipationFitLeastSquares(
        x=args[1], y=args[2], n=args[3], Amp0=args[4], gamma0=args[5], relNoise=args[6])
    dfls.run(disp=True)

if ('__main__' == __name__):
    main(sys.argv)
