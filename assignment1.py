"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random
import tqdm

class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """
        self.arr_off_all_bezier_t=[]
        self.arr_of_x_points=[]
        pass

    """ int this function we want to choose our point what we will work with to interpulate f"""

    def makePointsForInterpulation(self, f, start, stop, n):
        linespaceArray = np.linspace(start, stop, num=n, endpoint=True)
        myPoints = []
        for i in range(n):
            x = linespaceArray[i]
            myPoints.append(np.array([x, f(x)]))
        return linespaceArray,myPoints

    """in this func we want to find the control point of the function to bulid the buizer3 i will do it withe the matrix 
    that we learn in the class"""

    def piecewise_bezier_interpolation_in_matrix(self, points):
        ai=self.get_ai(points)
        bi=self.get_bi(points,ai)
        return ai , bi



    def get_ai(self, points):
        """build the w matrix"""
        num_of_point_cotrol_point = len(
            points) - 1  # this value is the num of control points ai/bi ,ai=bi,only need n-1
        mtx = np.zeros((num_of_point_cotrol_point, num_of_point_cotrol_point))
        np.fill_diagonal(mtx, 4)
        mtx[0][0] = 2
        mtx[num_of_point_cotrol_point - 1][num_of_point_cotrol_point - 1] = 7
        np.fill_diagonal(mtx[0:, 1:], 1)
        np.fill_diagonal(mtx[1:, 0:], 1)
        mtx[num_of_point_cotrol_point - 1][num_of_point_cotrol_point - 2] = 2
        w = mtx
        a = []
        b = []
        c = []
        for i in range(1, len(w[0])):
            a.append(w[i][i - 1])
        for i in range(len(w[0])):
            b.append(w[i][i])
        for i in range(len(w[0]) - 1):
            c.append(w[i][i + 1])
        """build k vector"""
        k = np.array(points[0] + 2 * points[1])
        for i in (range(1, num_of_point_cotrol_point - 1)):
            k = np.vstack((k, 4 * points[i] + 2 * points[i + 1]))
        k = np.vstack((k, 8 * points[num_of_point_cotrol_point - 1] + points[num_of_point_cotrol_point]))
        return self.alg_solv_mtx(a, b, c, k)


    def get_bi(self,points,ai):
        bi=[]
        n=len(points)-1
        for i in range(n-1):
           bi.append(2*points[i+1] - ai[i+1])
        bi.append(ai[n-2]+4*ai[n-1]-4*points[n-1])
        return bi

    def alg_solv_mtx(self, f1, f2, f3, f4):
        nolf = len(f4)

        for it in range(1, nolf):
            self.exp_to(f1, f2, f3, f4, it)
        xc=f2
        xc[-1] = f4[-1] / f2[-1]
        for il in range(nolf - 2, -1, -1):
            xc[il] = self.calc_for_mtx_solv(f2, f3, f4, il, xc)
        return xc

    def calc_for_mtx_solv(self, f2, f3, f4, il, xc):
        return (f4[il] - f3[il] * xc[il + 1]) / f2[il]

    def exp_to(self, f1, f2, f3, f4, ets):
        mc = f1[ets - 1] / f2[ets - 1]
        f2[ets] = f2[ets] - mc * f3[ets - 1]
        f4[ets] = f4[ets] - mc * f4[ets - 1]

    def bezier3(self,P1, P2, P3, P4):
        M = np.array(
            [[-1, +3, -3, +1],
             [+3, -6, +3, 0],
             [-3, +3, 0, 0],
             [+1, 0, 0, 0]],
            dtype=np.float32
        )
        P = np.array([P1, P2, P3, P4], dtype=np.float32)

        def f(t):
            T = np.array([t ** 3, t ** 2, t, 1], dtype=np.float32)
            return T.dot(M).dot(P)

        return f


    def makeBezierCurve(self,points,ai,bi):
        n=len(ai)
        bezCurv=[]
        for i in range(n):
            bezCurv.append(self.bezier3(points[i],ai[i],bi[i],points[i+1]))
        return bezCurv


    def findt(self,x):
        arr_of_x_points=self.arr_of_x_points
        arr_of_t_functions=self.arr_off_all_bezier_t
        s=arr_of_x_points[0]
        e=arr_of_x_points[ len(arr_of_x_points)-1]
        if x > e or x < s :
            return None,-1
        i=0
        while not (s <= x < arr_of_x_points[i + 1]):
            s= arr_of_x_points[i+1]
            i=i+1


        if( i > len(arr_of_x_points)-1 ):
            return arr_of_x_points[len(arr_of_x_points)-1] , (len(arr_of_x_points)-1)
        return arr_of_t_functions[i] ,i


    def regularFalsi(self, f, x1, x2, err=0.001):
        xh = (x1 * f(x2) - x2 * f(x1)) / (f(x2) - f(x1))
        if f(x1) * f(x2) <= 0:
            while np.abs(f(xh)) < err:
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


    def function_i_want_to_return (self, x):
        bezier_t ,i = self.findt(x)
        if bezier_t == None:
            return self.arr_off_all_bezier_t[len(self.arr_off_all_bezier_t)-1](x)[1]


        f=lambda t: bezier_t(t)[0] - x
        treal = self.regularFalsi(f,0,1)
        return self.arr_off_all_bezier_t[i](treal)[1]




    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        onlyxpointx ,point = self.makePointsForInterpulation(f, a, b, n)
        self.arr_of_x_points=onlyxpointx
        ai ,bi = self.piecewise_bezier_interpolation_in_matrix(point)

        bezierCurv = self.makeBezierCurve(point,ai,bi )
        self.arr_off_all_bezier_t=bezierCurv


        # replace this line with your solution to pass the second test
        return  self.function_i_want_to_return



##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)
            err = err / 200
            mean_err += err
        mean_err = mean_err / 100
        T = time.time() - T
        print(T)
        print(mean_err)



    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
