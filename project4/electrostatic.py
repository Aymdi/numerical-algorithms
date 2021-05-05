from NR import *
from math import *
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as leg 


def electrostatic_energy(x):
    '''

    Returns the value of electrostatic energy of the system

    Parameters
    ----------

    x: Positions of the charges of type array

    Returns
    ----------

    out: the value of electrostatic energy of the system

    '''

    res = 0
    for i in range(len(x)):
        sum = 0
        for j in range(len(x)):
            if j != i:
                sum += log(abs(x[i]-x[j]))
        res += log(abs(x[i] + 1)) + log(abs(x[i] - 1)) + 0.5*sum
    return res


def gradientE(x):
    '''

    Returns the gradient vector of the electrostatic energy function

    Parameters
    ----------

    x: Positions of the charges of type array

    Returns
    ----------

    out: the gradient vector of the electrostatic energy function

    '''

    n = len(x)
    v = np.zeros(n)
    for i in range(n):
        somme = 0
        for j in range(n):
            if j != i:
                #somme += 1 / abs(x[i] - x[j])
                somme += 1 / (x[i] - x[j]) 
        v[i] = 1 / (x[i] + 1) - 1 / (1 - x[i]) + somme
    return v


def Jacob_gradientE(x):
    '''

    Returns the Jacobian matrix associated to the gradient vector of the electrostatic energy function

    Parameters
    ----------

    x: Positions of the charges of type array

    Returns
    ----------

    out: the Jacobian matrix of the gradient vector of the electrostatic energy function

    '''

    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                somme = 0
                for k in range(n):
                    if k != i:
                        somme += 1 / (x[i] - x[k])**2
                J[i][i] = -1 / (x[i] + 1)**2 - 1 / (x[i] - 1)**2 - somme
            else:
                  J[i][j] = 1 / (x[i] - x[j])**2
    return J


def resolve(U0, epsilon, N, a_min=0.001):
    '''

    Looks for the electrostatic equilibrium positions using Newton_Raphson method

    Parameters
    ----------

    U0: The starting position of the algorithm

    epsilon: The precision wanted

    N: The maximal number of iterations allowed

    Returns
    ----------

    out: the electrostatic equilibrium positions 

    '''
    
    return Newton_Raphson(gradientE, Jacob_gradientE, U0, epsilon, N, a_min)


def extremum(x):
    nil = np.zeros(len(x))
    if electrostatic_energy(x) >= electrostatic_energy(nil):
        print("Equilibrium positions correspond to a maximum of electrostatic energy: OK\n")
    else:
        print("Equilibrium positions correspond to a minimum of electrostatic energy: FAILED\n")


def test_equilibrium_positions(x,epsilon):
    if (abs(gradientE(x).all()) <= epsilon):
        print("The positions calculated correspond well to the equilibrium positions of the system: OK\n ")
    else:
        print("The positions calculated don't correspond to the equilibirium positions of the system: FAILED\n ")


def show_energy_1charge():
    fig=plt.figure()
    x = np.linspace(-0.9,0.9,100)
    y = [ (electrostatic_energy(np.array([i]))) for i in x ]
    plt.plot(x,y)
    plt.xlabel("Charge's position")
    plt.ylabel("Electrostatic energy")
    plt.title("Electrostatic energy of a system of one charge")
    plt.savefig("img/energy_1charge.png",dpi=fig.dpi)
    print("creating img/energy_1charge.png\n ")


def plot(U0, epsilon, N, a_min=0.001):
    fig=plt.figure()
    coeff=np.zeros(len(U0)+2)
    coeff[len(U0)+1]=1
    plt.plot(resolve(U0, epsilon, N, a_min),np.zeros((len(U0))),'+',color='b',label='Electrostatic equilibrium positions')
    plt.plot(leg.Legendre(coeff).deriv().roots(),np.zeros((len(U0))),'x',color='r',label='Legendre polynomial\'s derivative roots')
    plt.legend()
    plt.title("Roots of Legendre polynomial derivative of order {}\n and the electrostatic equilibrium's positions of the system".format(len(U0)))
    plt.savefig("img/roots_{}".format(len(U0)), dpi=fig.dpi)
    print("creating img/roots_{}\n".format(len(U0)))


def test_electrostatic():
    sol = resolve([0], 1e-15, 1000, 0.001)
    extremum(sol)
    test_equilibrium_positions(sol,1e-15)
    plot([-0.5, 0, 0.5, 0.7, 0.8, 0.9], 1e-15, 1000, 0.001)
    plot([-0.5,0,0.5], 1e-15, 1000, 0.001)
    show_energy_1charge()


##########################################################


if __name__ == '__main__':
    test_electrostatic()
