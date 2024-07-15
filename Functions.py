# Farzad Zandi, 2024.
# Benchmark Functions.
import math
import numpy as np
import pandas as pd

# F1.
def powell_sum(x):
    return np.sum(np.abs(x) ** (np.arange(len(x)) + 2))

# F2.
def cigar(x):
    cigar = x[0]**2 + 10**6 * np.sum(x[1:]**2)
    return cigar

# F3.
def discus(x):
    discus = 10**6 * x[0]**2 + np.sum(x[1:]**2)
    return discus

# F4.
def Rosenbrock(x):
    # x_shifted = np.roll(x, shift=1)
    # sum_term = np.sum(100 * (x_shifted - x ** 2) ** 2 + (1 - x) ** 2)
    sum_term = 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2
    return sum_term

# F5.
def Ackley(x):
    c = 2 * np.pi
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))
    term2 = -np.exp(0.5 * (np.cos(c*x[0]) + np.cos(c*x[1])))
    return term1 + term2 + 20 + np.exp(1)

# F6.
def weierstrass(x, kmax=20, a=0.5, b=3):
    n = len(x)
    ak = a ** np.arange(kmax)
    bk = b ** np.arange(kmax)
    a = np.sum([np.sum([ak[i] * np.cos(bk[i] * 2 * np.pi * (xi + 0.5)) for i in range(kmax)]) for xi in x])
    b = n * np.sum([ak[i] * np.cos(bk[i] * 2 * np.pi * 0.5) for i in range(kmax)])   
    return a - b

# F7.
def Griewank(x):
    if isinstance(x, (list, np.ndarray)):
        n = len(x)
    else:
        n = 1
    sum_part = np.sum(x**2 / 4000.0)
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    return 1 + sum_part - prod_part

# F8.
def Rastrigin(x):
    n = len(x)
    return 10*n + sum([(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])

# F9.
def modified_schwefel(x):
    n = len(x)
    z = x + 4.209687462275036E+002
    return 418.9829 * n - g(z)

def g(z):
    n = len(z)
    s = 0
    for i in range(n):
        if abs(z[i])<500:
            s += z[i] * np.sin(abs(z[i])**0.5)
        if z[i]>500:
            s += (500 - np.mod(z[i], 500)) * np.sin(np.sqrt(abs(500 - np.mod(z[i], 500)))) - (z[i] - 500)**2 / (10000*n)
        if z[i]<-500:
            s += (np.mod(abs(z[i]), 500) - 500) * np.sin(np.sqrt(abs(np.mod(abs(z[i]), 500) - 500))) - (z[i] + 500)**2 / (10000*n)
    return s

# F10.
def katsuura(x):
    n = len(x)
    product_term = 1
    for i in range(n):
        sum_term = np.sum([(abs(2**j * x[i] - round(2**j * x[i])) / (2**j)) for j in range(1, 33)]) 
        product_term *= (1 + (i+1) * sum_term) ** (10 / (n**1.2))
    return (10 / (n**2)) * product_term - (10 / (n**2))

# F11.
def happy_cat(x):
    n = len(x)
    term1 = abs(np.sum([xi**2 - n for xi in x])) ** 0.25
    term2 = (0.5 * np.sum(x**2) + np.sum(x)) / n + 0.5
    return term1 + term2

# F12.
def HGBat(x):
    n = len(x)
    term1 = abs(np.sum(x**2)**2 - np.sum(x)**2) ** 0.5
    term2 = (0.5 * np.sum(x**2) + np.sum(x)) / n + 0.5
    return term1 + term2

# F13.
def Expanded_Griwank(x):
    s = 0
    n = len(x)
    for i in range(n-1):
        s += Griewank(Rosenbrock(np.array([x[i], x[i+1]])))
    s += Griewank(Rosenbrock(np.array([x[n-1], x[0]])))
    return s

# F14.
def Expanded_Scaffer(x):
    s = 0
    n = len(x)
    for i in range(n-1):
        s += g14(x[i], x[i+1])
    s += g14(x[n-1], x[0])
    return s

def g14(x, y):
    return 0.5 + (np.sin(np.sqrt(x**2 + y**2))**2 - 0.5) / ((1 + 0.001*(x**2 + y**2))**2)

# F15.
def some_diff(x):
    n = len(x)
    k = 0.5
    return 1 - (np.sum([math.cos(k * xi) * math.exp((-xi**2)/2) for xi in x]) / n)

# F16.
def Sphere(x):
    return np.sum([xi**2 for xi in x])

# F17.
def u(x, a, k, m):
    y = k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < (-a))
    return y

def penalized(x):
    n = len(x)
    y = 1 + (x+1) / 4
    y = np.pi / n * (10 * np.sin(np.pi * y[0])**2 + np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:] )**2)) + (y[-1] - 1)**2) + np.sum(u(x, 10, 100, 4))
    return y 

# F18.
def penalized2(x):
    d = len(x)
    term1 = 0.1 * (np.sin(3 * np.pi * x[0]) ** 2 + np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2)) + (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2))
    term2 = np.sum(u(x, 5, 100, 4))
    return term1 + term2

# F19.
def quartic(x):
    n = len(x)
    return sum([(i+1) * x[i]**4 for i in range(n)]) + np.random.random()

# F20.
def schwefel_1_2(x):
    n = len(x)
    return sum([sum(x[:i+1])**2 for i in range(n)])

# F21.
def schwefel_2_21(x):
    return np.max(np.abs(x))

# F22.
def schwefel_2_22(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

# F23.
def step_2(x):
    return np.sum(np.floor(x + 0.5) ** 2)

# F24.
def alpine1(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

# F25.
def csendes(x):
    n = len(x)
    return np.sum([x[i]**6 * (2 + np.sin(1/x[i])) for i in range(n)])

# F26.
def Rotated_Ellipse(x):
    return 7 * x[0]**2 - 6 * np.sqrt(3) * x[0] * x[1] + 13 * x[1]**2

# F27.
def Rotated_Ellipse_2(x):
    return x[0]**2 - x[0] * x[1] + x[1]**2

# F28.
def schwefel_2_24(x):
    return np.sum((x-1)**2 + (x[0] - x**2)**2)

# F29.
def sum_squares(x):
    n = len(x)
    return sum([(i+1) * x[i]**2 for i in range(n)])

# F30.
def step(x):
    return np.sum(np.floor(np.abs(x)))

# F31.
def schwefel(x):
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# F32.
def Xin_She(x):
    n = len(x)
    return np.sum([np.random.random() * np.abs(x[i])**i for i in range(n)])

# F33.
def schaffer(x):
    return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2

# F34.
def sgn(x, y):
    # if x<0:
    #     return x * (-1)
    # else:
    #     return x
    return x * np.sign(x), y * np.sign(y)
 
# F35.
def adjiman(x):
    return math.cos(x[0]) * math.sin(x[1]) - x[0] / (x[1]**2 + 1)

# F36.
def bartels_conn(x):
    return abs(x[0]**2 + x[1]**2 + x[0]*x[1]) + abs(math.sin(x[0])) + abs(math.cos(x[1]))

# F37.
def Ackely2(x):
    return -200 * np.exp(-0.02 * np.sqrt(x[0]**2 + x[1]**2))

# F38.
def Eggcrate(x):
    return x[0]**2 + x[1]**2 + 25 * (np.sin(x[0])**2 + np.sin(x[1])**2)

# F39.
def f40(x):
    return x[0] * np.sin(4 * x[0]) + 1.1 * x[1] * np.sin(2 * x[1])

# F40.
def powell_singular_2(x):
    n = len(x)
    return np.sum([(x[i-1] + 10 * x[i]) ** 2 + 5 * (x[i+1] - x[i+2])**2 + (x[i] - 2*x[i+1])**4 + 10*(x[i-1] + x[i+2])**4 for i in range(1, n-2)])

# F41.
def quintic(x):
    n = len(x)
    return np.sum([np.abs(x[i]**5 - 3*x[i]**4 + 4*x[i]**3 + 2*x[i]**2 - 10*x[i] - 4) for i in range(n)])

# F42.
def Qing(x):
    n = len(x)
    return np.sum([(x[i]**2 - i)**2 for i in range(1, n)])

# F43.
def salomon(x):
    return 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x**2))) + 0.1 * np.sqrt(np.sum(x**2))

# F44.
def dixon_price(x):
    n = len(x)
    sum_term = sum([(i+1) * (2*x[i]**2 - x[i-1])**2 for i in range(1, n)])
    return (x[0] - 1)**2 + sum_term
