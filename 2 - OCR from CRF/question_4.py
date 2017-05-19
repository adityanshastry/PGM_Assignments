from scipy import optimize
import numpy as np


def f(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def main():
    opt = optimize.minimize(f, np.array([1, 1]))
    print 'Value of function at minimum: ', opt['fun']
    print 'Values of x and y for minimum', opt['x']
    return


if __name__ == '__main__':
    main()
