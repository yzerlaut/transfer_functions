"""
Robert Schmidt, Bartosz Telenczuk
Models of Neural Systems
Winter Term 2007/08

Solving Ordinary Differential Equations
"""

import unittest
from scipy import *

def euler(f_func, x_0, t_max, dt):
    """Euler method for solving systems of differential equations"""
    x = asarray(x_0)
    res = [x_0]
    for t in arange(dt,t_max,dt):
        dx = f_func(x, t)
        x = x + asarray(dx)*dt
        res.append(x)
    return array(res)

class ODETests(unittest.TestCase):
    def testEuler1(self):
        f_log = lambda x, t: x*(1-x)
        sol_log = lambda t: exp(t)/(1+exp(t))
        
        time, y = euler(f_log, .5, 10, 0.1)
        y_exact = sol_log(time)
        err= mean((y - y_exact)**2)
        
        self.failUnless(err < 0.01)
        
if __name__ == '__main__':
    unittest.main()
