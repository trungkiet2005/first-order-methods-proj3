import numpy as np

def sphere(x): return np.sum(x**2)
def grad_sphere(x): return 2 * x

def rosenbrock(x): return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
def grad_rosenbrock(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

def beale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
def grad_beale(x):
    x1, x2 = x[0], x[1]
    part1 = 2 * (1.5 - x1 + x1*x2) * (-1 + x2)
    part2 = 2 * (2.25 - x1 + x1*x2**2) * (-1 + x2**2)
    part3 = 2 * (2.625 - x1 + x1*x2**3) * (-1 + x2**3)
    dx1 = part1 + part2 + part3
    
    part1_y = 2 * (1.5 - x1 + x1*x2) * (x1)
    part2_y = 2 * (2.25 - x1 + x1*x2**2) * (2*x1*x2)
    part3_y = 2 * (2.625 - x1 + x1*x2**3) * (3*x1*x2**2)
    dx2 = part1_y + part2_y + part3_y
    return np.array([dx1, dx2])

