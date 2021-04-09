# -*- coding: utf-8 -*-

import numpy as np
from random import randint

def calc_rand_non_zero() :

    '''
    calc_rand_non_zero(...)

    returns a random non zero number 

    Parameters
    ----------

    None

    Returns
    -------

    out : returns a random non zero number 

    '''
    
    i = randint(1,50)
    signe = 2*(0.5-randint(0,1))
    return signe*i
        
def generate_SPDSM(n,i):

    '''
    generate_SPDSM(...)

    returns a symetrical positive definite sparse matrix of nXn dim with i non zero extra-diagonal coefficents

    Parameters
    ----------

    n : positive integer wich is the size of the square matrix

    i : even number, between 0 and n(n-1), of non zero extra-diagonal coefficients

    Returns
    -------

    out : a symetrical positive definite sparse matrix of nXn dim with i non zero extra-diagonal coefficents


    '''
    
    i = i - i%2

    A = np.zeros([n,n])

    while np.count_nonzero(A) < i :

        line = randint(1,n-1)
        col = randint(0,line-1)
            
        if A[line,col] == 0 :
            A[line,col] = calc_rand_non_zero()
            A[col,line] = A[line,col]

    for i in range(n) :

        A[i,i] = abs(calc_rand_non_zero()*10)+ 10*sum(abs(A[i,:]))

    return A
        
        

if __name__=="__main__" :

    function_name = ["calc_rand_non_zero","generate_SPDSM"]
    test_result = {}
    number_of_test = {}

    for name in function_name :
        test_result[name]=0
        number_of_test[name]=0

    def print_summary(function_name,test_result,number_of_test):
        print(6*"-","SUMMARY",6*"-")
        for name in function_name :
            if test_result[name] == number_of_test[name] :
                test = '\033[32mPASSED\033[0m'
            else :
                test = '\033[31mFAILED\033[0m'
            print("{} : {}({}/{})".format(name,test,test_result[name],number_of_test[name]))
        return None



    
    print("MODULE : PARTIE_1")
    print("error_threshold for tests : 10**(-6)")
    print("structural_test__calc_rand_non_zero : ")

    nb_test = 22
    for i in range(nb_test) :
        f = calc_rand_non_zero
        x = f()
        if x != 0 :
            test_result[f.__name__]+=1
        number_of_test[f.__name__]+=1
            
    print("structural_test__generate_SPDSM : ")

    n = 5
    i = 2
    f = generate_SPDSM
    A = f(n,i)
    check = True
    
    for j in range(n) :
        if A[j,j] <= (sum(abs(A[j,:]))-A[j,j]) :
            check = False
            
    if len(A) == n and np.all(np.diag(A)>0) and np.count_nonzero(A-np.diag(np.diag(A))) == i and check:
        test_result[f.__name__]+=1
    number_of_test[f.__name__]+=1

    n = 15
    i = 6
    f = generate_SPDSM
    A = f(n,i)
    check = True

    for j in range(n) :
        if A[j,j] <= (sum(abs(A[j,:]))-A[j,j]) :
            check = False
    if len(A) == n and np.all(np.diag(A)>0) and np.count_nonzero(A-np.diag(np.diag(A))) == i and check:
        test_result[f.__name__]+=1
    number_of_test[f.__name__]+=1

    n = 7
    i = 4
    f = generate_SPDSM
    A = f(n,i)
    check = True

    for j in range(n) :
        if A[j,j] <= (sum(abs(A[j,:]))-A[j,j]) :
            check = False
    if len(A) == n and np.all(np.diag(A)>0) and np.count_nonzero(A-np.diag(np.diag(A))) == i and check:
        test_result[f.__name__]+=1
    number_of_test[f.__name__]+=1
    
    print_summary(function_name,test_result,number_of_test)
