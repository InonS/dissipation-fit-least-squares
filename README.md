# dissipation-fit-least-squares
Fit a first-order (exponential) decay to a signal using scipy.optimize.minimize

## Motivation
In an attempt to find a Python (preferably SciPy) package to fit non-linear curves with constraints, I had no luck finding an example that implements simple regression analysis. 
In my particular case, I only needed this for an exponentially decaying function with two parameters: Amplitude and extinction coefficient. Furthermore, I impose conditions only on the bounds of the parameters, and not more complex constraints (as in the more general form of the Linear Programming problem).
My target function (to be minimized) is non-linear, in particular I'm doing an Ordinary Least Squares regression. This was the example I was looking for. I hope it may serve others, too.

## Introduction
I've invested some time in commenting the code and adding logging printouts, to hopefully assist whoever is trying to follow along.
The module can be run as a stand-alone program (e.g. from the command-line or Python interpreter), or imported and called from user code (please cite your source if you do).
The code is organized into a class (DissipationFitLeastSquares) and two global methods (exponentialDecay and main). The program is used in two steps:
1. Creating a DissipationFitLeastSquares object, and passing in various construction arguments (such as the input signal to be fitted, and initial guesses for the parameters, etc.)
2. Calling DissipationFitLeastSquares::run().

