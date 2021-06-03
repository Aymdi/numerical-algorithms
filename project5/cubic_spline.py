import numpy as np
import matplotlib.pyplot as plt
import math
from load_foil import load_foil
import sys


def spline(X, Y):
    n = len(Y)
    Y2 = [0] * n
    U = [0] * (n-1)

    for i in range(1, n-1):
        sig = (X[i] - X[i-1]) / (X[i+1] - X[i-1])
        p = sig * Y2[i-1] + 2
        Y2[i] = (sig - 1) / p
        U[i] = (Y[i+1] - Y[i]) / (X[i+1] - X[i]) - (Y[i] - Y[i-1]) / (X[i] - X[i-1])
        U[i] = (6 * U[i] / (X[i+1] - X[i-1]) - sig * U[i-1])/p

    for k in range(n-2,0,-1):
        Y2[k]=Y2[k]*Y2[k+1]+U[k]

    return Y2


def splint(X,Y,x):
    n = len(Y)
    klo = 0
    khi = n-1
    Y2 = spline(X,Y)

    while(khi-klo > 1):
        k = (khi+klo) // 2
        if X[k] > x:
            khi = k
        else:
            klo = k

    h = X[khi] - X[klo]
    if h == 0:
        raise Exception("Bad X input to routine splint")
    a = (X[khi] - x)/h
    b = (x - X[klo])/h
    return a * Y[klo] + b * Y[khi] + ((a**3 - a) * Y2[klo] + (b**3 - b) * Y2[khi]) * h**2 / 6

def splint_der(X,Y,x):
    n = len(Y)
    klo = 0
    khi = n-1
    Y2 = spline(X,Y)

    while(khi-klo > 1):
        k = (khi+klo) // 2
        if X[k] > x:
            khi = k
        else:
            klo = k

    h = X[khi] - X[klo]
    if h == 0:
        raise Exception("Bad X input to routine splint")
    a = (X[khi] - x)/h
    b = (x - X[klo])/h
    return (Y[khi] - Y[klo]) / h - (3 * a**2 - 1)/6 * h * Y2[klo] + (3 * b**2 - 1)/6 * h * Y2[khi]

def get_foil(a,b,N,file):
        
    (dim,ex,ey,ix,iy) = load_foil( file )

    pas = (b-a)/N

    ex2 = [a+pas*k for k in range (N+1)]
    ey2 = [splint(ex,ey,x) for x in ex2]

    ix2 = [a+pas*k for k in range (N+1)]
    iy2 = [splint(ix,iy,x) for x in ix2]

    return ex2,ey2,ix2,iy2

def test_splint():
    functions = [
        math.exp,
        #math.sin,
        math.atan,
        lambda x: x * math.sin(x),
        math.tan,
        math.log,
        math.sqrt
    ]

    labels = [
        "Exponentielle",
        #"Sinus",
        "Arc tangente",
        "x --> x * sin(x)",
        "Tangente",
        "Logarithme népérien",
        "Racine carrée"
    ]

    plt.ylabel("Ecart relatif entre la fonction d'interpolation\net la fonction à interpoler (en norme infinie)")
    plt.xlabel("Nombre de points d'interpolation")
    plt.yscale("log")
    #plt.title("Ecarts relatifs d'interpolation de différentes fonctions usuelles,\nen fonction du nombre de points d'interpolation")
    
    for k in range(len(functions)):
        f = functions[k]
        n = 50
        I = np.linspace(.1, math.pi/2 - .1, 1000)
        X = list(range(2, n))
        Y = [0] * (n-2)
        for i in range(2, n):
            X_ = np.linspace(.1, math.pi/2 - .1, i)
            Y[i-2] = max([abs(splint(X_,[f(x) for x in X_],x) - f(x)) for x in I]) / max([abs(f(x)) for x in I])
        plt.plot(X, Y, label=labels[k])
    plt.legend()
    plt.show()

def test_splint_der():
    functions = [
        math.exp,
        #math.sin,
        math.atan,
        lambda x: x * math.sin(x),
        math.tan,
        math.log,
        math.sqrt
    ]
    derivatives = [
        math.exp,
        #math.cos,
        lambda x: 1 / (1 + x**2),
        lambda x: math.sin(x) + x * math.cos(x),
        lambda x: 1 / math.cos(x)**2,
        lambda x: 1/x,
        lambda x: 1/(2 * math.sqrt(x))
    ]

    labels = [
        "Exponentielle",
        #"Sinus",
        "Arc tangente",
        "x --> x * sin(x)",
        "Tangente",
        "Logarithme népérien",
        "Racine carrée"
    ]
    
    plt.ylabel("Ecart relatif entre les dérivées de la fonction\nd'interpolation et la fonction à interpoler (en norme infinie)")
    plt.xlabel("Nombre de points d'interpolation")
    plt.yscale("log")
    #plt.title("Ecarts relatifs d'interpolation de différentes fonctions usuelles, en\nfonction du nombre de points d'interpolation (comparaison des dérivées)")
    
    for k in range(len(functions)):
        f = functions[k]
        f_ = derivatives[k]
        n = 50
        I = np.linspace(.1, math.pi/2 - .1, 1000)
        X = list(range(2, n))
        Y = [0] * (n-2)
        for i in range(2, n):
            X_ = np.linspace(.1, math.pi/2 - .1, i)
            Y[i-2] = max([abs(splint_der(X_,[f(x) for x in X_],x) - f_(x)) for x in I]) / max([abs(f_(x)) for x in I])
        plt.plot(X, Y, label=labels[k])
    plt.legend()
    plt.show()

def test_foil():
    
    file = "foil.txt"
    (dim,ex,ey,ix,iy) = load_foil( file )
    (ex2,ey2,ix2,iy2) = get_foil(0,1,100,file)

    plt.axis('equal')
    plt.plot(ex,ey,'o')
    plt.plot(ix,iy,'o')
    plt.plot(ex2,ey2)
    plt.plot(ix2,iy2)

    plt.show()


if __name__ == "__main__":
    test_splint()
    test_splint_der()
    test_foil()






    
