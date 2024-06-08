"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random
from assignment2 import Assignment2
from commons import f1, f2, f3, f6, f10


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    def sympson(self,f, a, b, n):
        h = (b - a) / n
        s = f(a) + f(b)
        for i in range(1, n, 2):
            s =s+ 4 * f(a  + i * h)
        for i in range(2, n - 1, 2):
            s =s+ 2 * f(a  + i * h)
        return s * h / 3


    def trapezoidal(self, f, a, b, n):
        return (b - a) * ((f(a) + f(b)) / 2)

    def myinteg(self, f, a, b, n):
        if n == 1:
            h = (a - b) / 2
            return 2 * h * f(a + h)
        if n == 2:
            return self.trapezoidal(f, a, b, n)
        if (n % 2 == 1):
            return self.sympson(f, a, b, n - 1)
        else:
            return self.sympson(f, a, b, n - 2)

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        return np.float32(self.myinteg(f, a, b, n))

    def newfunc (self,f1,f2):
        return lambda y: f2(y)-f1(y)


    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution
        start =1
        end=100
        myass = Assignment2()
        points = myass.intersections(f1, f2, start, end)
        lenp=len(points)
        if lenp < 2 :
            return np.float32('nan')
        newFunctionForIntegration = self.newfunc(f1,f2)
        returnval=0
        for i in range(lenp-1):
            returnval=returnval+(np.abs(self.integrate(newFunctionForIntegration,points[i],points[i+1],100)))
        return np.float32(returnval)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
import matplotlib.pyplot as plt


class TestAssignment3(unittest.TestCase):

    def test_a(self):
        a=np.linspace(1,100,10000)
        yf1=[]
        for i in a:
            yf1.append(f2(i))
        yf2 = []
        for i in a:
            yf2.append(f2(i))


        plt.plot(a,yf1)

        plt.plot(a,yf2)
        plt.show()
        ass=Assignment3()
        aa=ass.areabetween(f10,f2)
        print(aa)


    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)
        print(ass3.integrate(f1, 0, 2, 10))
        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        self.my()
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_my(self):
        print(222)
        ass = Assignment3()
        f1 = np.poly1d([0.3, -2, 0, 1])
        f2 = np.poly1d([1, -4])
        print(ass.areabetween(f1, f2))



if __name__ == "__main__":
    unittest.main()
