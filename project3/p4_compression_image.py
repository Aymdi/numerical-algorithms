#Partie 4 - Question 1 - compression au rang k

import numpy as np
import math
import matplotlib.pyplot as plt
from p2_bidiagonale import *
from p3_transfo_QR import *
from tests import *


def compress_k(A,k):
    '''
    returns the matrix after k-rank compression 

    Parameters
    -----------
    A : a matrix of type array
    k : an integer

    Returns
    ----------
    output : A matrix of type array

    '''
    (L,BD,R) = bidiagonalize(A)
    (U,S,V) = transfo_QR(BD,50)
    (U,S) = modifyUS(U,S)
    for i in range(k, len(S)):
        S[i][i]=0

    return L.dot(U.dot(S.dot(V.dot(R))))


def compress_img(img_name, k):
    img_full = plt.imread("img/"+img_name)
    A1 = take_nth_pixel(img_full, 0)
    A2 = take_nth_pixel(img_full, 1)
    A3 = take_nth_pixel(img_full, 2)

    A1 = compress_k(A1,k)
    A2 = compress_k(A2,k)
    A3 = compress_k(A3,k)

    (p, q, t) = np.shape(img_full)
    for i in range(p):
        for j in range(q):
            # values less than 0 displayed in red
            if (A1[i][j] < 0.0) :
                A1[i][j] = 1.0
            if (A2[i][j] < 0.0) :
                A2[i][j] = 0.0
            if (A3[i][j] < 0.0) :
                A3[i][j] = 0.0

            # values more than 1 displayed in green
            if (A1[i][j] > 1.0) :
                A1[i][j] = 0.0
            if (A2[i][j] > 1.0) :
                A2[i][j] = 1.0
            if (A3[i][j] > 1.0) :
                A3[i][j] = 0.0

            img_full[i][j][0] = A1[i][j]
            img_full[i][j][1] = A2[i][j]
            img_full[i][j][2] = A3[i][j]

    plt.imsave("img/result_"+img_name,img_full)

def p4_tests():
    function_name = ["compress_k"]
    (test_result, number_of_test) = init_tests(function_name)

    ###################################

    print("MODULE : PARTIE_4")

    print("error_threshold for tests : 10**(-6)")

    print("test__compress_k : ")

    A = np.array([
        [8,0,0,0],
        [0,5,0,0],
        [0,0,3,0],
        [0,0,0,2]])
    k = 2

    hope = np.array([
        [8,0,0,0],
        [0,5,0,0],
        [0,0,0,0],
        [0,0,0,0]])

    res = compress_k(A, k)
    main_test("compress_k", [A,k], res, hope, test_result, number_of_test)

    ###################################

    print_summary(function_name,test_result,number_of_test)

    print()
    print("Compression de l'image essai.png au rang 100 dans result_essai.png")
    compress_img("essai.png", 100)

if __name__=="__main__" :
    p4_tests()
