"""
In this assignment you should fit a model function of your choice to data
that you sample from a contour of given shape. Then you should calculate
the area of that shape.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you know that your iterations may take more
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment.
Note: !!!Despite previous note, using reflection to check for the parameters
of the sampled function is considered cheating!!! You are only allowed to
get (x,y) points from the given shape by calling sample().
"""

import numpy as np
import time
import random

from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.special import comb

from functionUtils import AbstractShape
import numpy as np
import matplotlib.pyplot as plt



class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self):
        self.xs=[]
        self.ys=[]
        pass
    def area(self) -> np.float32:
        ass5=Assignment5()
        return ass5.calcAreaPoly(self.xs,self.ys,len(self.xs))
    def sample(self):
        pass
    def contour(self, n: int):
        pass

class Assignment5:
    def __init__(self):

        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """
        pass


    def iterForCalc(self, contour, maxerr):
        numOfPoints = 100
        retVal = 0
        points = contour(numOfPoints)
        X = points.T[0]
        Y = points.T[1]
        area =self.calcAreaPoly(X,Y,len(points))
        dif = abs(area-retVal)
        while dif > maxerr and numOfPoints < 10000:
            retVal = area
            numOfPoints = numOfPoints * 2
            points = contour(numOfPoints)
            X = points.T[0]
            Y = points.T[1]
            area = self.calcAreaPoly(X,Y,len(points))
            dif = abs(retVal - area)
        return area


    def calcAreaPoly(self, X, Y, numOfPoints):
        ar = 0.0
        j = numOfPoints - 1
        for i in range(numOfPoints):
            ar += (X[j]*(Y[j] - Y[i]) + X[i]*(Y[j] - Y[i]))
            j = i
        return abs(ar) / 2


    def area(self, contour: callable, maxerr=0.001) -> np.float32:

        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        re = self.iterForCalc(contour, maxerr)
        return np.float32(re)
    """
    sort arr of points according to the cordinates
    """
    def NormalizePtc(self, ptc, points):
        ptc[0] = ptc[0] / len(points)
        ptc[1] = ptc[1] / len(points)
        return ptc

    def sortArrOfPints(self, points):
        ptc = [0, 0]
        for i in points:
            ptc[0] = ptc[0] + i[0]
            ptc[1] = ptc[1] + i[1]
        ptc = self.NormalizePtc(ptc, points)
        for i in points:
            a = i[0] - ptc[0]
            i[0] = a
            b = i[1] - ptc[1]
            i[1] = b
        pointsRet = sorted(points, key=self.cmp)
        for i in pointsRet:
            a = i[0] + ptc[0]
            i[0] = a
            b = i[1] + ptc[1]
            i[1] = b
        return pointsRet
    def getAngle(self, ptc, pt):
        x = pt[0] - ptc[0]
        y = pt[1] - ptc[1]
        ang = np.arctan2(y, x)
        if ang <= 0:
            ang = 2 * np.pi + ang
        return ang

    def getDistanse(self, pt1, pt2):
        x = pt1[0] - pt2[0]
        y = pt1[1] - pt2[1]
        return np.sqrt(x * x + y * y)

    def cmp(self, x):
        return (self.getAngle([0, 0], x), self.getDistanse([0, 0], x))

    def comparePoints(self, p1, p2):
        an1 = self.getAngle([0, 0], p1)
        an2 = self.getAngle([0, 0], p2)
        if an1 < an2:
            return True
        d1 = self.getDistanse([0, 0], p1)
        d2 = self.getDistanse([0, 0], p2)
        if an1 == an2 and d1 < d2:
            return True
        return False

    def piontsForMY(self, num, sample):
            x = num
            myPoints = []
            while x > 0:
                z,t=sample()
                myPoints.append([z,t])
                x -= 1
            return myPoints




    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """
        num=maxtime*145
        rs=MyShape()
        points=np.array(self.piontsForMY(num, sample))
        points=self.sortArrOfPints(points)
        points = np.array(points).T
        points[0][num - 1] = points[0][0]
        points[1][num - 1] = points[1][0]
        t, u = interpolate.splprep(points, u=None, s=1.9, per=True)
        un = np.linspace(u.min(), u.max(), 10000)
        xs, ys = interpolate.splev(un, t,der=0)
        rs.ys=ys
        rs.xs=xs
        plt.show()
        return rs






##########################################################################



import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_circle_area8(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        counter = 0
        for i in range(100):
            shape = ass5.fit_shape(sample=circ, maxtime=30)
            a = shape.area()
            if np.abs(a - np.pi) >0.01:
                print(a-np.pi)
                counter += 1
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)



    def test_circledsd_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        T = time.time() - T
        counter = 0
        print(1)

        for i in range(100):
            print(1)
            shape = ass5.fit_shape(sample=circ, maxtime=30)
            a = shape.area()
            if np.abs(a - np.pi) < 0.01:
                counter += 1

        print(counter, "000000000000000")
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
