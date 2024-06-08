"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def trySquizeRange(self, a, b,f, maxIter):
        control = False
        for i in range(maxIter):
            if f(a) * f(b) <= 0:
                return a, b
            else:
                if control:
                    a = a + ((abs(b - a)) / 10)
                    control = False
                else:
                    b = b + -((abs(b - a)) / 10)
                    control = True

        return None, None

    def regularFalsi(self, f, x1, x2, err, maxfpos=500):
        xh = 0
        if f(x1) * f(x2) <= 0:
            for fpos in range(1, maxfpos + 1):
                if (f(x1) == 0):
                    return x1
                if (f(x2) == 0):
                    return x2
                xh = (x1 * f(x2) - x2 * f(x1)) / (f(x2) - f(x1))
                if np.abs(f(xh)) < err:
                    return xh
                elif f(x1) * f(xh) < 0:
                    x2 = xh
                else:
                    x1 = xh
        else:
            return None
        if (np.abs(f(xh)) > err):
            return None
        return xh

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution

        fnew = lambda x: f1(x) - f2(x)
        lenspace=np.abs(b-a)
        arrayOfPoints = np.linspace(a, b, 120, endpoint=True)
        retArr = []
        for i in range(len(arrayOfPoints) - 1):
            a = arrayOfPoints[i]
            b = arrayOfPoints[i + 1]
            if fnew(a) * fnew(b) > 0:  # two points with the same sign
                a, b = self.trySquizeRange(a, b,fnew, 100)
                if a != None and b != None and fnew(a) * fnew(b) <= 0 :
                    regularFalsiRes = self.regularFalsi(fnew, a, b, maxerr)
                    if regularFalsiRes != None:
                        retArr.append(regularFalsiRes)
            else:
                regularFalsiRes = self.regularFalsi(fnew, a, b, maxerr)
                if regularFalsiRes != None:
                    retArr.append(regularFalsiRes)
        # replace this line with your solution
        return retArr


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def my(self):
        ass2 = Assignment2()
        f1 = np.poly1d([1.53, 1.134, 0.6208, 0.5356, - 1.171, - 0.1678, - 0.8692, - 1.291, 0.05584, 0.4915, - 0.6076])
        f2 = np.poly1d([-0.6595, - 0.6723, - 0.196, - 1.611, 0.5696, -0.8761, 0.2914, 1.82, 2.087, - 0.03512, 0.6344])
        f1 = lambda x: np.sin(x)
        f2 = lambda x: np.power(x - 2, 2)
        X = ass2.intersections(f1, f2, 0, 100, maxerr=0.001)
        print(X)




    def test_sqr(self):
        self.my()


    def test_poly(self):
        ass2 = Assignment2()
        for i in range(100):
            f1, f2 = randomIntersectingPolynomials(10)
            X = ass2.intersections(f1, f2, -10, 10, maxerr=0.001)
            print(X)
            for x in X:
                self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))



if __name__ == "__main__":
    unittest.main()
