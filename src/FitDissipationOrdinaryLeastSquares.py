'''
@author: Inon Sharony
@contact: www.tau.ac.il/~inonshar
@organization: Tel Aviv University Department of Chemical Physics
@since:  Aug 2, 2015
'''
from argparse import ArgumentParser
from logging import basicConfig, info, debug
from math import exp
from matplotlib.pyplot import plot, show
from numpy import linspace, array, square, var
from random import gauss
from scipy.optimize import minimize


class FitDissipationOrdinaryLeastSquares():
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
    STDEV[eps] = Amp / SNR

    parameters:
    thetas = [ A, g ]

    target function:
    function(t;A,g) = Sum over t of [ ( y(t) - f(t;A,g) )^2 ]
                           = Sum over t of [ y^2 -2*y*f + f^2 ]

    target Jacobian (for GGA methods. Not necessary, but speeds convergence up):
    jacobian(t;A,g) = Sum over t of [ d(function(t;A,g)) / dA , d(function(t;A,g)) / dg ]

    target Hessian (for meta-GGA methods. not necessary, but speeds convergence up):
    hessian(t;A,g) = Sum over t of
    [ [ d^2(function(t;A,g)) / dA^2 , d^2(function(t;A,g)) / dAdg ],
      [ d^2(function(t;A,g)) / dgdA , d^2(function(t;A,g)) / dg^2 ] ]
    """

    @classmethod
    def exponentialDecay(cls, t, Amp, gamma):
        return Amp * exp(-gamma * t)

    def __init__(self, x=None, y=None, Amp0=1.0, gamma0=1.0, SNR=10.0):

        basicConfig(level="DEBUG")

        if x is not None and y is not None and array(x).shape != array(y).shape:
            raise Exception(
                "data features and labels cannot have mismatching dimensions! x : {} != {} : y".format(x.shape, y.shape))

        N = 1000
        self.x = array(x if x is not None else linspace(
            0, 1, y.shape[0] if y is not None else N))
        self.n = self.x.shape[0]

        self._A_ = Amp0
        self._g_ = gamma0
        self.dof = self.n - len([self._A_, self._g_])

        self.y = array(
            y if y is not None else self.generateNoisySignal(SNR))

        info("{}\nn = {}\nx = {}\ny = {}\nAmp0 = {}\ngamma0 = {}\nSNR = {}"
             .format(self, self.n, self.x.shape, self.y.shape, self._A_, self._g_, SNR))

        self.varData = var(self.y)
        self.SStot = self.varData * self.n
        debug("(Var[y] = {:.5g}) = ({:s} = {:.5g}) / (dof = {:d})"
              .format(self.varData, r'${SS}_{tot}$', self.SStot, self.dof))

    def generateNoisySignal(self, SNR):
        """
        fx = [exponentialDecay(t, self._A_, self._g_) for t in self.x]
        rx = fx + gauss(0, self._A_ / SNR)
        """
        info(
            "generating noisy signal from {} * exp (-{} * t) + N(0, {})".format(self._A_, self._g_, self._A_ * SNR))
        return [(FitDissipationOrdinaryLeastSquares.exponentialDecay(t, self._A_, self._g_) +
                 gauss(0, self._A_ / SNR)) for t in self.x]

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
        R2 = 1 - SSres / self.SStot
        info("coefficient of determination ({:s}) = {:.2%}"
             .format(r'$R^2$', R2))
        # chi^2 >> 1 : under-fitting, chi^2 << 1 : over-fitting
        Chi2 = SSres / self.varData
        info("goodness of fit ({:s}, {:s}{:d}) = {:.2%}".format(
            r'$\Chi^2_{\ni}$', r'$\ni=$', self.dof, Chi2))

        A, g = optimizeResult.x
        plot(self.x, self.y, 'r.', self.x,
             [FitDissipationOrdinaryLeastSquares.exponentialDecay(t, A, g) for t in self.x], 'b_')
        show()
        return optimizeResult, R2, Chi2


def main(args):
    dfls = FitDissipationOrdinaryLeastSquares(
        args.x, args.y, args.Amp0, args.gamma0, args.SNR)
    dfls.run(disp=True)

if ('__main__' == __name__):
    parser = ArgumentParser(
        description='Computes effective dissipation coefficient\
        for an exponentially decaying (scalar) signal\
        with random Gaussian Markovian noise.')
    parser.add_argument(
        '-x', type=list, help='list of time points')
    parser.add_argument(
        '-y', type=list, help='values of measurement at time points')
    parser.add_argument(
        '-Amp0', type=float, help='initial guess for dissipation amplitude', default=1.0)
    parser.add_argument(
        '-gamma0', type=float, help='initial guess for extinction coefficient', default=10.0)
    parser.add_argument(
        '-SNR', type=float, help='signal to noise ratio', default=25)

    parser.print_help()

    main(parser.parse_args())
