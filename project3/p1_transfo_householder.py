#Partie1: Transformations de Housholder

import numpy as np
import math
from tests import *


def symmetry_vector(X, Y):
    '''

    returns the symmetry vector so that its Householder matrix returns X over Y
    
    Parameters
    -----------
    X : the first vector of type array
    Y : the second vector of type array

    Returns
    ----------
    output : A vector of type array

    '''
    if ((X==Y).all()) :
        return np.zeros_like(X)
    return (X-Y)/np.linalg.norm(X-Y)


def HouseholderXY(X,Y):
    '''
    
    returns the symmetry matrix that returns X over Y
    
    Parameters
    -----------
    X : the first vector of type array
    Y : the second vector of type array

    Returns
    ----------
    output : Householder matrix 

    '''
    sizeH = X.shape[0]
    U = symmetry_vector(X, Y)
    return (np.identity(sizeH) - 2 * np.dot(U,U.T))


def Householder(U):
    '''
    
    returns the symmetry matrix from its symmetry vector
    
    Parameters
    -----------
    U : a vector of type array

    Returns
    ----------
    output : Householder matrix 

    '''
    return (np.identity(np.size(U)) - 2 * np.dot(U,U.T))


##########################################################################


def scalar_product(X, Y):
    '''
    
    returns the scalar product of two vectors
    
    Parameters
    -----------
    X : an horizental vector 
    Y : a vertical vector

    Returns
    ----------
    output : the scalar product

    Examples
    ----------
    scalar_product([[1,1]],[[1],[1]]) = 2 

    '''
    size = Y.shape[0]
    res = 0
    for i in range(size):
        res += X[0][i] * Y[i][0]
    return res


def Householder_product_vector(U,v):
    '''
    
    returns the product of the Householder matrix of the symetric vector U by the vector v
    
    Parameters
    -----------
    U : a symmetric matrix
    v : a vertical vector

    Returns
    ----------
    output : the product in a vertical vector shape

    '''
    return v - 2 * U * scalar_product(U.T,v)


def list_to_array(L):
    '''
    
    converts a list to an a vertical vector of type array

    '''
    size = L.size
    res = np.zeros(shape=(size,1))
    for i in range(3):
        res[i] = L[i]
    return res


def Householder_product_matrix(U,M):
    '''
    
    returns the product of the Householder matrix of the symetric vector U by a matrix M
    
    Parameters
    -----------
    U : a symmetric matrix
    M : the second matrix

    Returns
    ----------
    output : the product

    '''
    n = U.shape[0]
    m = M.shape[1]
    P = np.zeros(shape=(n,m))
    for j in range(m):
        values = Householder_product_vector(U,list_to_array(M[:,j]))
        for i in range(values.size):
            P[i][j] = values[i]
    return P


def calculate_ij_product(H, v, i, j):

    sizeH = H.shape[1]

    sum = 0
    for x in range(sizeH):
        sum += H[i][x] * v[x][j]

    return sum


def householder_matrix_dot_product(H, v):
    sizeH = H.shape[0]
    sizeV = v.shape[1]

    matrix = np.zeros((sizeH,sizeV))

    for i in range(sizeH):
        for j in range(sizeV):
            matrix[i][j] = calculate_ij_product(H, v, i, j)

    return matrix

def p1_tests():
    print("MODULE : PARTIE_1")

    print("error_threshold for tests : 10**(-6)")

    print("test__symmetry_vector : ")

    function_name = ["symmetry_vector","HouseholderXY","Householder","scalar_product","Householder_product_vector","Householder_product_matrix"]
    (test_result, number_of_test) = init_tests(function_name)

    #TEST1
    X = np.array([[1],[0],[0]])
    Y = np.array([[0],[1],[0]])
    res = symmetry_vector(X,Y)
    hope = np.array([[1/math.sqrt(2)],[-1/math.sqrt(2)],[0]])
    main_test("symmetry_vector",[X,Y],res,hope, test_result, number_of_test)
    #TEST2
    X = np.array([[2],[0],[0]])
    Y = np.array([[0],[1],[0]])
    res = symmetry_vector(X,Y)
    hope = np.array([[2/math.sqrt(5)],[-1/math.sqrt(5)],[0]])
    main_test("symmetry_vector",[X,Y],res,hope, test_result,number_of_test)
    #TEST3
    X = np.array([[2],[0],[2]])
    Y = np.array([[0],[1],[0]])
    res = symmetry_vector(X,Y)
    hope = np.array([[2/3],[-1/3],[2/3]])
    main_test("symmetry_vector",[X,Y],res,hope, test_result,number_of_test)

    ###################################

    print("test__HouseholderXY")

    #TEST1
    X = np.array([[3],[4],[0]])
    Y = np.array([[0],[0],[5]])
    res = HouseholderXY(X,Y)
    hope = np.array([[0.64,-0.48,0.6],[-0.48,0.36,0.8],[0.6,0.8,0]])
    main_test("HouseholderXY",[X,Y],res,hope, test_result, number_of_test)

    ###################################

    print("test__Householder")

    #TEST1
    U = np.array([[1],[1],[1]])
    res = Householder(U)
    hope = np.array([[-1,-2,-2],[-2,-1,-2],[-2,-2,-1]])
    main_test("Householder",[U],res,hope, test_result, number_of_test)
    #TEST2
    U = np.array([[0],[0],[0]])
    res = Householder(U)
    hope = np.array([[1,0,0],[0,1,0],[0,0,1]])
    main_test("Householder",[U],res,hope, test_result, number_of_test)

    ###################################

    print("test__scalar_product :")

    #TEST1
    X = np.array([[1,1]])
    Y = np.array([[1],[1]])
    res = scalar_product(X,Y)
    hope = 2
    main_test("scalar_product",[X,Y],res,hope, test_result, number_of_test)
    #TEST2
    X = np.array([[9,1,0]])
    Y = np.array([[1],[2],[5]])
    res = scalar_product(X,Y)
    hope = 11
    main_test("scalar_product",[X,Y],res,hope, test_result, number_of_test)

    ###################################

    print("test__Householder_product_vector")

    #TEST1
    v = np.array([[1],[1],[1]])
    U = np.array([[0],[0],[0]])
    res = Householder_product_vector(U,v)
    hope = np.array([[1],[1],[1]])
    main_test("Householder_product_vector",[U,v],res,hope, test_result, number_of_test)
    #TEST2
    v = np.array([[1],[1],[1]])
    U = np.array([[1],[1],[1]])
    res = Householder_product_vector(U,v)
    hope = np.array([[-5],[-5],[-5]])
    main_test("Householder_product_vector",[U,v],res,hope, test_result, number_of_test)

    ###################################

    print("Householder_product_matrix")

    #TEST1
    U = np.array([[9],[10],[11]])
    M = np.array([[2,0,2],[0,0,0],[2,0,0]])
    res = Householder_product_matrix(U,M)
    hope = np.dot(Householder(U),M)
    main_test("Householder_product_matrix",[U,M],res,hope, test_result, number_of_test)
    #TEST2
    U = np.array([[1],[2],[3],[4]])
    M = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    res = Householder_product_matrix(U,M)
    hope = np.dot(Householder(U),M)
    main_test("Householder_product_matrix",[U,M],res,hope, test_result, number_of_test)

    ###################################

    print_summary(function_name,test_result,number_of_test)


if __name__=="__main__" :
    p1_tests()
    