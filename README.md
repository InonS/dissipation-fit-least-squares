# dissipation-fit-least-squares
Fit a first-order (exponential) decay to a signal using scipy.optimize.minimize

[![DOI](https://zenodo.org/badge/18810/InonS/dissipation-fit-least-squares.svg)](https://zenodo.org/badge/latestdoi/18810/InonS/dissipation-fit-least-squares)

![Dissipation Fit Least Squares](resources/DissipationFitLeastSquares.jpg)

## Motivation
In an attempt to find a Python (preferably SciPy) package to fit non-linear curves with constraints, I had no luck finding an example that implements simple regression analysis. 

In my particular case, I only needed this for an exponentially decaying function with two parameters: Amplitude and extinction coefficient. Furthermore, I impose conditions only on the bounds of the parameters, and not more complex constraints (as in the more general form of the Linear Programming problem).

My target function (to be minimized) is non-linear, in particular I'm doing an Ordinary Least Squares regression. This was the example I was looking for. I hope it may serve others, too.

## Introduction
I've invested some time in commenting the code and adding logging printouts, to hopefully assist whoever is trying to follow along.

The module can be run as a stand-alone program (e.g. from the command-line or Python interpreter), or imported and called from user code (please cite your source if you do).

The code is organized into a class (FitDissipationOrdinaryLeastSquares) and a main() function. The program is used in two steps:

1. Creating a FitDissipationOrdinaryLeastSquares object, and passing in various construction arguments (such as the input signal to be fitted, and initial guesses for the parameters, etc.)

2. Calling FitDissipationOrdinaryLeastSquares.run().

## Development notes
I've tried generalizing this to any function (using SymPy) but found that multidimensional derivatives (function with respect to unknown parameters) are hard to implement with SymPy. It's just not worth the bother; I suggest migrating such problems to Julia or Sage, for example.

Forking and pull requests are welcome! (please cite your source if you do)

## Contact
[Inon Sharony](www.tau.ac.il/~inonshar)
