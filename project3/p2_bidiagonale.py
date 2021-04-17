import matplotlib.pyplot as plt
import numpy as np
from p1_transfo_householder import *
from tests import *


def take_nth_pixel(M, n):
    (p, q, t) = np.shape(M)
    D = np.zeros((p, q))
    for i in range(p):
        for j in range(q):
            D[i][j] = M[i][j][n]
    return D

def bidiagonalize(A):
    (n, m) = A.shape
    Qleft = np.identity(n)
    Qright = np.identity(m)
    BD = A.copy()
    for i in range(min(n,m)):
        v1 = np.reshape((np.zeros(n - i)), (n-i, 1))
        v1[0] = np.linalg.norm(BD[i:n, i])
        
        Q1 = HouseholderXY(np.reshape(BD[i:n, i], (n-i, 1)), v1)  #        Let Q1 be a HH matrix mapping BD[i:n,i] on a vector with a single non-zero element
        Qleft[i:n, i:n] = Qleft[i:n, i:n].dot(Q1)               #        Qleft = Qleft * Q1
        BD[i:n, i:m] = Q1.dot(BD[i:n, i:m])                     #        BD    = Q1 * BD,
        
        if (i <= (m-2)):                                # If (not(i==(m-2)))
            v2 = np.reshape((np.zeros(m-(i+1))), (m-(i+1), 1))
            v2[0] = np.linalg.norm(BD[i, (i+1):m])
            
            Q2 = HouseholderXY(np.reshape(BD[i, (i+1):m], (m-i-1, 1)), v2) #         Let Q2 be a HH matrix mapping BD[i,(i+1):m] on a vector with a single non-zero element
            Qright[(i+1):m, (i+1):m] = Q2.dot(Qright[(i+1):m, (i+1):m])             #            Qright = Q2 * Qright
            BD[i:n, (i+1):m] = BD[i:n, (i+1):m].dot(Q2)                         #        BD     = BD * Q2
        

    return (Qleft, BD, Qright)

def p2_tests():

    print("MODULE : PARTIE_2")

    function_name = ["bidiagonale"]
    (test_result, number_of_test) = init_tests(function_name)

    img_full = np.array(plt.imread("img/essai.png"))
    A1 = take_nth_pixel(img_full, 0)
    (QL1, BD1, QR1) = bidiagonalize(A1)
    res = QL1.dot(BD1.dot(QR1))
    print(res.shape)
    hope = A1
    print(hope.shape)
    main_test("bidiagonale", [A1], res, hope, test_result, number_of_test)

    ###################################

    print_summary(function_name,test_result,number_of_test)


if __name__=="__main__" :

    p2_tests()

