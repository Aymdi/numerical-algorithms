## Functions
import numpy as np
from random import randint
import time as tmp

import sys
# sys.path.append("/home/tim/Documents/ENSEIRB/1A/6/algo_num/proj2/partie_2/../partie_1") # Ignore that line
sys.path.append("../partie_1")

from partie1_q1 import calc_ti
from partie1_q1 import calc_tji

from partie1_q3 import generate_SPDSM
from partie1_q3 import calc_rand_non_zero

from partie1_q4 import facto_cholesky_incomplete
from partie1_q4 import facto_cholesky_incomplete_REC


def prod(A,B) :
    '''
    prod(...)

    returns the transpose of a matrix.

    Parameters
    ----------

    A : a matrix of dimension n,m
    B : a matrix of dimension n',m'


    Returns
    -------

    out : The product of the two given matrix

    Time complexity
    ---------------

    O(n*m')

    Space complexity
    ----------------

    O(1)

    '''
    return A.dot(B)

def trans(A) :
    '''
    trans(...)

    returns the transpose of a matrix.

    Parameters
    ----------

    A : a matrix

    Returns
    -------

    out : the transpose of the matrix

    Time complexity
    ---------------

    O(1)

    Space complexity
    ----------------

    O(1)

    '''
    return A.transpose()


def conjugate_gradient(A,b,x) :
    '''
    conjugate_gradient(...)

    returns the solution to the equation Ax = b

    Parameters
    ----------

    A : positive definite square matrix
    b : vector solution of the equation Ax = b
    x : initial estimation of the searched vector

    Returns
    -------

    out : The vector x, solution to the equation Ax = b

    Time complexity
    ---------------

    inferior to n**3

    Space complexity
    ----------------

    O(n)

    '''

    r = b-prod(A,x)                         # The inverse of the gradient of f
    p = r                                   # First direction of the being built base
    rsold = prod(trans(r),r)

    for i in range (1,10**6) :
        Ap = prod(A,p)
        alpha = rsold/(prod(trans(p),Ap))   # coordinate of the solution in the base
        x = x+alpha*p                       # Variable that converges towards the solution
        r = r-alpha*Ap                      # new gradient
        rsnew = prod(trans(r),r)
        if np.sqrt(rsnew[0][0]) < 10**(-10) :
            break

        p = r+(rsnew/rsold)*p               # new direction
        rsold = rsnew
    return x

def triangular_solve_down(A,b) :
    '''
    triangular_solve_down(...)

    returns the solution to the equation Ax = b where A is a lower triangular matrix

    Parameters
    ----------

    A : a lower triangular matrix
    b : a column vector solution of the equation Ax = b

    Returns
    -------

    out : The vector x, solution to the equation Ax = b

    Time complexity
    ---------------

    O(n**2)

    Space complexity
    ----------------

    O(n)

    '''

    n = len(A)

    x = np.zeros((n,1))

    for ligne in range(n) :
        somme = b[ligne][0]
        for colonne in range(ligne) :
            somme -= A[ligne][colonne]*x[colonne]
        x[ligne] = somme/A[ligne][ligne]
    return x



def triangular_solve_up(A,b) :
    '''
    triangular_solve_up(...)

    returns the solution to the equation Ax = b where A is a upper triangular matrix

    Parameters
    ----------

    A : a upper triangular matrix
    b : a column vector solution of the equation Ax = b

    Returns
    -------

    out : The vector x, solution to the equation Ax = b

    Time complexity
    ---------------

    O(n**2)

    Space complexity
    ----------------

    O(n)

    '''
    n = len(A)

    x = np.zeros((n,1))

    for ligne in range(n-1,-1,-1) :
        somme = b[ligne][0]
        for colonne in range(n-1,ligne,-1) :
            somme -= A[ligne][colonne]*x[colonne]
        x[ligne] = somme/A[ligne][ligne]
    return x


def solve(T,Tt,r) :
    '''
    solve(...)

    returns the solution x to the equation T*Tt*x = b

    Parameters
    ----------

    T : a lower triangular matrix
    Tt : a upper triangular matrix
    b : vector solution of the equation T*Tt*x = b
    x : initial estimation of the searched vector

    Returns
    -------

    out : The vector x, solution to the equation T*Tt*x = b

    Time complexity
    ---------------

    O(n**2)

    Space complexity
    ----------------

    O(n)

    '''
    y = triangular_solve_down(T,r)
    z = triangular_solve_up(Tt,y)
    return z


def conjugate_gradient_preconditioned(A,b,x) :
    '''
    conjugate_gradient_preconditioned(...)

    returns the solution to the equation Ax = b

    Parameters
    ----------

    A : positive definite square matrix
    b : vector solution of the equation Ax = b
    x : initial estimation of the searched vector

    Returns
    -------

    out : The vector x, solution to the equation Ax = b

    Time complexity
    ---------------

    inferior to n**3

    Space complexity
    ----------------

    O(n**2)

    '''
    r = b-prod(A,x)                         # The inverse of the gradient of f
    T = facto_cholesky_incomplete(A)        # Incomplete Cholesky preconditionner
    Tt = trans(T)

    z = solve(T,Tt,r)
    p = z                                   # First direction of the being built base
    rsold = prod(trans(r),z)

    for i in range (1,10**6) :
        Ap = prod(A,p)
        alpha = rsold/(prod(trans(p),Ap))   # Coordinate of the solution in the base
        x = x+alpha*p                       # Variable that converges towards the solution
        r = r-alpha*Ap                      # New gradient
        z = solve(T,Tt,r)
        rsnew = prod(trans(r),z)
        if np.sqrt(rsnew[0][0]) < 10**(-10) :
            break

        p = z+(rsnew/rsold)*p               # New direction
        rsold = rsnew
    return x



## End of program's function
if __name__ == "__main__" :
## Execution of all algorithms

    n = 10

    A = generate_SPDSM(n,20)
    print(A)

    b = np.ones((n,1))
    x = np.zeros((n,1))

    sol = conjugate_gradient(A,b,x)
    print(sol)
    print("\n",np.linalg.solve(A,b))

    sol2 = conjugate_gradient_preconditioned(A,b,x)
    print("\n",sol2)

    ## Tests

    print("MODULE : PARTIE_2")
    print("error_threshold for tests : 10**(-6)")
    print("\nstructural_test__conjugate_gradient : ")

    threshold = 10**(-6)

    A = generate_SPDSM(5,10)
    b = np.array([[1.0],[2.0],[3.0],[4.0],[5.0]])
    x = np.zeros((5,1))

    x = conjugate_gradient(A,b,x)

    res = prod(A,x)

    flag = 0
    for k in range (len(A)) :
        if (np.absolute(res[k][0]-b[k][0]) > threshold) :
            flag += 1

    if (flag == 0) :
        print('Test conjugate_gradient : \033[32mPASSED\033[0m')
    else :
        print('Test conjugate_gradient : \033[31mFAILED\033[0m')


    print("\nstructural_test__triangular_solve_down : ")

    T = facto_cholesky_incomplete(A)
    x = triangular_solve_down(T,b)

    res = prod(T,x)

    flag = 0
    for k in range (len(T)) :
        if (np.absolute(res[k][0]-b[k][0]) > threshold) :
            flag += 1

    if (flag == 0) :
        print('Test triangular_solve_down : \033[32mPASSED\033[0m')
    else :
        print('Test triangular_solve_down : \033[31mFAILED\033[0m')


    print("\nstructural_test__triangular_solve_up : ")

    Tt = trans(T)
    x = triangular_solve_up(Tt,b)

    res = prod(Tt,x)

    flag = 0
    for k in range (len(Tt)) :
        if (np.absolute(res[k][0]-b[k][0]) > threshold) :
            flag += 1

    if (flag == 0) :
        print('Test triangular_solve_up : \033[32mPASSED\033[0m')
    else :
        print('Test triangular_solve_up : \033[31mFAILED\033[0m')


    print("\nstructural_test__conjugate_gradient : ")

    x = conjugate_gradient_preconditioned(A,b,x)

    res = prod(A,x)

    flag = 0
    for k in range (len(A)) :
        if (np.absolute(res[k][0]-b[k][0]) > threshold) :
            flag += 1

    if (flag == 0) :
        print('Test conjugate_gradient_preconditioned : \033[32mPASSED\033[0m')
    else :
        print('Test conjugate_gradient_preconditioned : \033[31mFAILED\033[0m')



    ## Time graph
    import matplotlib.pyplot as plt


    def test() :
        time = []
        result_sans_pre = []
        result_avec_pre = []
        for N in range(2,600,2) :
            coef = N/20
            time += [N]
            A = generate_SPDSM(N,coef)
            b = np.ones((N,1))
            x = np.zeros((N,1))

            print("sans prec")

            tmp1=tmp.perf_counter()
            sol = conjugate_gradient(A,b,x)
            tmp2=tmp.perf_counter()
            result_sans_pre += [tmp2-tmp1]

            print("avec prec")


            tmp1=tmp.perf_counter()
            sol = conjugate_gradient_preconditioned(A,b,x)
            tmp2=tmp.perf_counter()
            result_avec_pre += [tmp2-tmp1]
        return time,result_sans_pre ,result_avec_pre



    x,y1,y2 = test()


    fig = plt.subplot()

    plt.grid(linestyle='--')
    fig.set_xlabel("Dimension de la matrice carré")
    fig.set_ylabel("Temps d'exécution de l'algorithme en secondes")
    plt.plot(x,y1,label="Méthode du gradient conjugué")
    plt.plot(x,y2,label="Méthode du gradient conjugué préconditionné")
    plt.legend()



    plt.show()

    ## iteration graph
    import matplotlib.pyplot as plt


    def conjugate_gradient(A,b,x) :
        r = b-prod(A,x)
        p = r
        rsold = prod(trans(r),r)

        for i in range (1,10**6) :
            Ap = prod(A,p)
            alpha = rsold/(prod(trans(p),Ap))
            x = x+alpha*p
            r = r-alpha*Ap
            rsnew = prod(trans(r),r)
            if np.sqrt(rsnew[0][0]) < 10**(-10) :
                break

            p = r+(rsnew/rsold)*p
            rsold = rsnew
        return x,i


    def conjugate_gradient_preconditioned(A,b,x) :
        r = b-prod(A,x)
        T = facto_cholesky_incomplete(A)
        Tt = trans(T)

        z = solve(T,Tt,r)
        p = z
        rsold = prod(trans(r),z)

        for i in range (1,10**6) :
            Ap = prod(A,p)
            alpha = rsold/(prod(trans(p),Ap))
            x = x+alpha*p
            r = r-alpha*Ap
            z = solve(T,Tt,r)
            rsnew = prod(trans(r),z)
            if np.sqrt(rsnew[0][0]) < 10**(-10) :
                break

            p = z+(rsnew/rsold)*p
            rsold = rsnew
        return x,i

    def test() :
        time = []
        result_sans_pre = []
        result_avec_pre = []
        for N in range(2,600,2) :
            coef = N/40
            time += [N]
            A = generate_SPDSM(N,coef)
            b = np.ones((N,1))
            x = np.zeros((N,1))

            print("sans prec")

            sol,i = conjugate_gradient(A,b,x)
            result_sans_pre += [i]

            print("avec prec")


            sol,i = conjugate_gradient_preconditioned(A,b,x)
            result_avec_pre += [i]
        return time,result_sans_pre ,result_avec_pre

    x,y1,y2 = test()


    fig = plt.subplot()

    plt.grid(linestyle='--',)
    fig.set_xlabel("Dimension de la matrice carré")
    fig.set_ylabel("Nombre d'itérations")
    plt.plot(x,y1,label="Méthode du gradient conjugué")
    plt.plot(x,y2,label="Méthode du gradient conjugué préconditionné")
    plt.plot(x,x,label="Nombre d'itération maximum de l'algorithme' (y=N)")
    plt.legend()

    plt.show()










