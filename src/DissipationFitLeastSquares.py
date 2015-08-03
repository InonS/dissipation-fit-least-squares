'''
Created on Aug 2, 2015

@author: Inon Sharony
'''
from logging import basicConfig, info, debug
from math import exp
from matplotlib.pyplot import plot, show
from numpy import linspace, array, square, var
from random import gauss
from scipy.optimize import minimize


def exponentialDecay(t, Amp, gamma):
    return Amp * exp(-gamma * t)


class DissipationFitLeastSquares():
    """
    Computes effective dissipation coefficient
    for an exponentially decaying (scalar) signal
    with random Gaussian Markovian noise
    """

    def __init__(self, n=1000, x=(0, 1), y=None, Amp0=1.0, gamma0=10.0, relNoise=0.04):

        basicConfig(level="DEBUG")

        info("{}\nn = {}\nx = {}\ny = {}\nAmp0 = {}\ngamma0 = {}\nrelNoise = {}"
             .format(self, n, x, y, Amp0, gamma0, relNoise))

        self.n = n
        self.x = linspace(x[0], x[1], self.n)

        self._A_ = Amp0
        self._g_ = gamma0
        self.dof = self.n - len([self._A_, self._g_])

        self.y = y if y is not None else self.generateNoisySignal(relNoise)

        self.varData = var(self.y)
        self.SStot = self.varData * self.dof
        debug("(Var[y] = {:.5g}) = ({:s} = {:.5g}) / (dof = {:d})"
              .format(self.varData, r'${SS}_{tot}$', self.SStot, self.dof))

    def generateNoisySignal(self, relNoise):
        self._eps_ = relNoise
        self.fx = [exponentialDecay(t, self._A_, self._g_) for t in self.x]
        self.rx = [(exponentialDecay(t, self._A_, self._g_) +
                    gauss(0, self._A_ * self._eps_)) for t in self.x]
        return array(self.rx)

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

        self.optimizeResult = minimize(fun=self.function,
                                       x0=array([self._A_, self._g_]),
                                       method="SLSQP", jac=self.jacobian,
                                       bounds=[(0, None), (0, None)],
                                       options={'disp': disp})

        info(repr(self.optimizeResult))

        self.SSres = self.optimizeResult.get('fun')
        # $R^2 \rightarrow 1^{-}$ : perfect fit, $R^2 \rightarrow 0^{+}$ : no fit
        info("coefficient of determination ({:s}) = {:.2%}"
             .format(r'$R^2$', 1 - self.SSres / self.SStot))
        # chi^2 >> 1 : under-fitting, chi^2 << 1 : over-fitting
        info("goodness of fit ({:s}, {:s}{:d}) = {:.2%}".format(
            r'$\Chi^2_{\ni}$', r'$\ni=$', self.n, self.SSres / self.varData))

        A, g = self.optimizeResult.x
        plot(self.x, self.y, 'r.', self.x,
             [exponentialDecay(t, A, g) for t in self.x], 'b--')
        show()


def main():
    dfls = DissipationFitLeastSquares()
    dfls.run(disp=True)

if ('__main__' == __name__):
    main()
