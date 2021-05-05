import numpy as np
from NR import Newton_Raphson
import math


def remainder(B, C, coefP):
    """
    This function computes the coefficients of the polynomial remainder after dividing the polynomial P(x) by the quadratic polynomial x^2 + Bx + C such that :
    P(x) = Q(x)(x^2 + Bx +C) + Rx + S

    Parameters :
    -----------
    B: an array.
        the coefficient of degree 1 in the quadratic polynomial.
    C: an array
        the coefficient of degree 0 in the quadratic polynomial.
    coefP: an array 
        array representing the coefficients of the polynomial P.

    Return :
    -------
    R : an array.
        the coefficient of degree 1 in the remainder polynomial.
    S : an array.
        the coefficient of degree 0 in the remainder polynomial.
    """
    if len(coefP.shape) == 1:
        div = np.polydiv(coefP, [1, B, C])[1]
        if len(div) == 1:
            return [0., div[0]]
        return div

    n = np.size(B)
    coef = np.ones((coefP.shape[0]))
    R = np.ones(n)
    S = np.ones(n)

    for i in range(n):
        b = B[i]
        c = C[i]
        q = np.array([1, b, c])
        # coeficient of the jth column of polynomial P
        coef = coefP[:, i:i+1].flatten()
        # remainder of the ith column of polynomial P
        R[i], S[i] = np.polydiv(coef, q)[1]
    return R, S


def remainder_test():
    # P = (x^2 + 2*x + 1)*x^4 + 8*x + 9
    # P = x^6 + 2*x^5 + x^4 + 8*x + 9
    P = np.array([1, 2, 1, 0, 0, 8, 9])
    v1_remainder = remainder(2, 1, P)
    assert np.array_equal(v1_remainder, [8, 9]), "should be [8, 9]"
    print("test remainder with vector of dimension 1 : OK")

    # P2 = (((1,1)x^2 + (5,2)x + (4,7))*x^2 + (5,8)x + (1, 5)
    P2 = np.array([[1, 1], [5, 2], [4, 7], [5, 8], [1, 5]])
    v2_remainder = remainder([5., 2.], [4., 7.], P2)
    assert np.array_equal(v2_remainder, [[5, 8], [1, 5]]), "should be [[5, 8], [1, 5]]"
    print("test remainder with vector of dimension 2 : OK")

    #P3 = [(1,2,3,4)x^5 + (2,5,8,6)x^4 + (5,2,4,7)x^3 +(5,2,8,4)][(1,1,1,1)x^2 + (5,3,8,9)x + (5,2,8,1)] + (5,2,8,4)x + (2,5,7,6)
    P3 = np.array([
                  [1, 2, 3, 4],
                  [7, 11, 32, 42],
                  [20, 21, 92, 97],
                  [35, 16, 96, 117],
                  [25, 4, 32, 63],
                  [5, 2, 8,  4],
                  [30, 8, 72, 40],
                  [27, 9, 71, 42]])
    v4_remainder = remainder([5, 3, 8, 9], [5, 2, 8, 9], P3)
    assert np.array_equal(v4_remainder, [[5, 2, 8, 4], [2, 5, 7, 6]]), "should be [[5, 2, 8, 4], [2, 5, 7, 6]]"
    print("test remainder with vector of dimension 4 : OK")
    print()


######################### Quotient #########################

def quotient(B, C, coefP):
    """Computes the quotient Q(x) of the polynomial division of P(x) by x²+ Bx+ C such that :
    P(x) = Q(x)(x^2 + Bx +C) + Rx + S

    Parameters :
    -----------
    B: an array.
        the coefficient of degree 1 in the quadratic polynomial.
    C: an array
        the coefficient of degree 0 in the quadratic polynomial.
    coefP: an array 
        array representing the coefficients of the polynomial P.

    Return :
    -------
    quot : an array.
        the coefficient of the polynomial representing the division of P(x) by a quadratic function.
    """
    if len(coefP.shape) == 1:
        return np.array(np.polydiv(coefP, [1, B, C])[0])

    n = np.size(B)
    P_degree = coefP.shape[0]
    coef = np.ones((P_degree))
    quot = np.ones((P_degree-2, n))

    for i in range(n):
        b = B[i]
        c = C[i]
        q = np.array([1, b, c])
        # coeficient of the jth column of polynomial P
        coef = coefP[:, i:i+1].flatten()
        # coefficient of the quotient of the jth row
        quot_j = np.polydiv(coef, q)[0]
        # resulting quotient polynomial
        quot[:, i:i+1] = [[x] for x in quot_j]
    return quot


def test_quotient():
    # P1 = ((1,1)x^2 + (5,2)x + (4,7))*[x^2] + (5,8)x + (1, 5)
    P1 = np.array([[1, 1], [5, 2], [4, 7], [5, 8], [1, 5]])
    quotient_P1 = quotient([5., 2.], [4., 7.], P1)
    assert np.array_equal(quotient_P1, [[1, 1], [0, 0], [0, 0]])
    print("test quotient with vector of dimension 2 : OK")

    #P2 = ([1,1,1]x² + [2,3,0]x +[1,1,1])(([1,1,1]x² +[2,3,0]x +[1,1,1])[1,1,1]x² + [2,3,5]x + [1,1,1]) + [4,2,1]x² + [0,7,2]
    #P2 = [1,1,1]x⁶ + [4,6,0]x⁵ + [6,11,2]x⁴ + [6,9,5]x³ + [10,13,3]x² + [4,6,5]x + [1,8,3]
    P2 = np.array([[1, 1, 1],
                    [4, 6, 0],
                    [6, 11, 2],
                    [6, 9, 5],
                    [10, 13, 3],
                    [4, 6, 5],
                    [1, 8, 3]])
    quotient_P2 = quotient([2, 3, 0], [1, 1, 1], P2)
    assert np.array_equal(quotient_P2, [[1, 1, 1], [2, 3, 0], [
                            1, 1, 1], [2, 3, 5], [5, 3, 2]])
    print("test quotient with vector of dimension 3 : OK")

    #P3 = [(1,2,3,4)x^5 + (2,5,8,6)x^4 + (5,2,4,7)x^3 + (5,2,8,4)] * [(1,1,1,1)x^2 + (5,3,8,9)x + (5,2,8,1)] + (5,2,8,4)x + (2,5,7,6)
    P3 = np.array([
                    [1, 2, 3, 4],
                    [7, 11, 32, 42],
                    [20, 21, 92, 65],
                    [35, 16, 96, 69],
                    [25, 4, 32, 7],
                    [5, 2, 8,  4],
                    [30, 8, 72, 40],
                    [27, 9, 71, 42]])
    quotient_P3 = quotient([5, 3, 8, 9], [5, 2, 8, 1], P3)
    assert np.array_equal(quotient_P3, [[1, 2, 3, 4], [2, 5, 8, 6], [
                            5, 2, 4, 7], [0, 0, 0, 0], [0, 0, 0, 0], [5, 2, 8, 4]])
    print("test quotient with vector of dimension 4 : OK")
    print()


######################### Partial derivative #########################

def partial_derivative_RS_C(B, C, coefP):
    """Computes the partial derivative of R and S by C such that :
  

    Parameters :
    -----------
    B: an array.
        the coefficient of degree 1 in the quadratic polynomial.
    C: an array
        the coefficient of degree 0 in the quadratic polynomial.
    coefP: an array 
        array representing the coefficients of the polynomial P.

    Return :
    -------
    -R1 : an array.
        the coefficient of degree 1 in partial derivatives of R.
    -S1 : an array.
        the coefficient of degree 0 in partial derivatives of S.
    """
    Q = quotient(B, C, coefP)
    R1, S1 = remainder(B, C, Q)
    return -R1, -S1


def test_partial_derivative_RS_C():
    # P = (x² + 2x +1)((x² + 2x + 1)*x³ + 4x + 2 ) + x +3
    # P = x⁷ + 4x⁶+ 6x⁵+4x⁴+5x³+10x²+9x+5
    P = np.array([1, 4, 6, 4, 5, 10, 9, 5])
    # the result must be -4, -2
    partial_derivative_P1 = partial_derivative_RS_C(2, 1, P)
    assert partial_derivative_P1 == (-4, -2)
    print("test partial derivative RS_C with vector of dimension 1 : OK")

    #P2 = ([1,1,1]x² + [2,3,0]x +[1,1,1])(([1,1,1]x² +[2,3,0]x +[1,1,1]) * ([1,1,1]x² + [2,3,5]x + [1,1,1]) +[4,2,1]x + [0,7,2]
    #P2 = [1,1,1]x⁶ + [4,6,0]x⁵ + [6,11,2]x⁴ + [6,9,5]x³ + [10,13,3]x² + [4,6,5]x + [1,8,3]
    P2 = np.array([[1, 1, 1],
                    [4, 6, 0],
                    [6, 11, 2],
                    [6, 9, 5],
                    [6, 11, 2],
                    [8, 8, 6],
                    [1, 8, 3]])
    partial_derivative_P2 = partial_derivative_RS_C([2, 3, 0], [1, 1, 1], P2)
    assert np.array_equal(partial_derivative_P2[0], [-2, -3, -5]) and np.array_equal(partial_derivative_P2[1], [-1, -1, -1])
    print("test partial derivative RS_C with vector of dimension 3 : OK")
    print()


def partial_derivative_RS_B(B, C, coefP):
    """Computes the partial derivatives of R and S by B such that :
    

    Parameters :
    -----------
    B: an array.
        the coefficient of degree 1 in the quadratic polynomial.
    C: an array
        the coefficient of degree 0 in the quadratic polynomial.
    coefP: an array 
        array representing the coefficients of the polynomial P.

    Return :
    -------
    -B * R1 - S1 : an array.
        the coefficient of the partial derivatives
    C * R1 : an array.
        the coefficient of partial derivatives
    """
    R1 = -partial_derivative_RS_C(B, C, coefP)[0]
    S1 = -partial_derivative_RS_C(B, C, coefP)[1]
    return B * R1 - S1, C * R1


def test_partial_derivative_RS_B():
    #(-1,1)
    P = np.array([1, 3, 6, 5, 8])
    partial_derivative_P1 = partial_derivative_RS_B(1, 1, P)
    assert np.array_equal(partial_derivative_P1, [-1, 1])
    print("test partial derivative RS_B with vector of dimension 1 : OK")


######################### quadric solution #########################

def quadric_solution(B, C):
    delta = B**2 - 4*C
    if delta >= 0:
        return [(-B - math.sqrt(delta))/2, 0], [(-B + math.sqrt(delta))/2, 0]
    else:
        return [-B/2, math.sqrt(-delta)/2], [-B/2, -math.sqrt(-delta)/2]


def test_quadric_solution():
    # x²+x+1
    print(quadric_solution(2, 1))


######################### Bairstow #########################

def bairstow(P):
    """the bairstow algorithm computes the zeros of a polynomial using the Newton Raphson method.
    
    Parameters:
    ----------
        P : polynomial
    
    Return :
    ------- 
        A list of the zeros of the polynomial P   
    """

    i = 0
    degree = P.size - 1
    solutions = np.zeros((degree, 2))

    while degree > 2:
        def f(U):
            return remainder(U[0], U[1], P)

        def J(U):
            return [partial_derivative_RS_B(U[0], U[1], P)[0], partial_derivative_RS_C(U[0], U[1], P)[0]], [partial_derivative_RS_B(U[0], U[1], P)[1], partial_derivative_RS_C(U[0], U[1], P)[1]]
        U0 = np.ones(2)
        B, C = Newton_Raphson(f, J, U0, 0.001, 500)
        solutions[i], solutions[i+1] = quadric_solution(B, C)

        i += 2
        degree -= 2
        P = quotient(B, C, P)

    # Case degree == 1
    if degree == 1:
        solutions[i] = [-P[1]/P[0], 0]
    # Case degree == 2
    if degree == 2:
        delta = P[1]**2 - 4*P[0]*P[2]
        if delta >= 0:
            X1 = [(-P[1] - math.sqrt(delta)) / (2*P[0]), 0]
            X2 = [(-P[1] + math.sqrt(delta)) / (2*P[0]), 0]
        else:
            X1 = [-P[1]/2, math.sqrt(-delta)/2]
            X2 = [-P[1]/2, -math.sqrt(-delta)/2]
        solutions[i] = X1
        solutions[i + 1] = X2

    return solutions


def test_bairstow():
    P = np.array([1, -2, -1, 2])
    # 1, -1, 2
    print("solution for P\n", bairstow(P))
    print()
    P1 = np.array([1, -1, -6, 0])
    # 3, 0, -2
    print("solution for P1\n", bairstow(P1))
    print()


def tests_bairstow():
  remainder_test()
  test_quotient()
  test_partial_derivative_RS_C()
  test_partial_derivative_RS_B()
  test_quadric_solution()
  test_bairstow()


###################################################################


if __name__ == '__main__':
  tests_bairstow()