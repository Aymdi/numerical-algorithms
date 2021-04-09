# -*- coding: utf-8 -*-

import numpy as np

def calc_ti(A,T,i) :

    '''
    calc_ti(...)

    returns the value of tii of the dense Cholesky factorization

    Parameters
    ----------

    A : positive definite square matrix

    T : the result of dense Cholesky factorization

    i : positive integer between 0 and the dimension of A-1

    Returns
    -------

    out : the value of tii

    Time complexity
    ---------------

    O(2i-1)

    Space complexity
    ----------------

    O(1)

    '''

    if i == 0 :
        return np.sqrt(A[i,i])
    return np.sqrt(A[i,i]-sum(T[i,:i]**2))

def calc_tji(A,T,i,j):

    '''
    calc_tji(...)

    returns the value of tji of the dense Cholesky factorization

    Parameters
    ----------

    A : positive definite square matrix

    T : the result of dense Cholesky factorization

    i : positive integer between 0 and the dimension of A-1

    j : positive integer between 0 and i

    Returns
    -------

    out : the value of tji

    Time complexity
    ---------------

    O(2i-1)

    Space complexity
    ----------------

    O(1)

    '''
    return (A[i,j]-sum(T[i,:i]*T[j,:i]))/T[i,i]

# vectorise_calc_tji = np.vectorize(calc_tji,excluded=[0,1,2],cache=True)

def facto_cholesky_dense_REC(A,T,i):

    '''
    facto_cholesky_dense_REC(...)

    completes T with the value of tii and tji for j from 0 to i-1 of the dense Cholesky factorization and returns a call of itself for next index

    Parameters
    ----------

    A : positive definite square matrix

    T : the result of dense Cholesky factorization

    i : positive integer between 0 and the dimension of A-1

    Returns
    -------

    out : itself with a call on next index

    Time complexity
    ---------------

    O((n-i+1)(2i-1))

    Space complexity
    ----------------

    O(1)

    '''

    if i == len(A) :
        return T
    T[i,i]=calc_ti(A,T,i)
        
    for j in range(i+1,len(A)) :
            
        T[j,i] = calc_tji(A,T,i,j)
    return facto_cholesky_dense_REC(A,T,i+1)
    
def facto_cholesky_dense(A) :

    '''
    facto_cholesky_dense(...)

    returns the matrix of the dense Cholesky factorization 

    Parameters
    ----------

    A : positive definite square matrix

    Returns
    -------

    out : the matrix of the dense Cholesky factorization

    Time complexity
    ---------------

    O(1/3*n**3)

    Space complexity
    ----------------

    O(n**2)

    '''

    T = np.zeros([len(A),len(A)])
    return facto_cholesky_dense_REC(A,T,0)

if __name__=="__main__" :

    function_name = ["calc_ti","calc_tji","facto_cholesky_dense_REC","facto_cholesky_dense"]
    test_result = {}
    number_of_test = {}

    for name in function_name :
        test_result[name]=0
        number_of_test[name]=0
        
    def main_test(name,entree,res,hope):

        error_threshold = 0.000001
        if type(hope)==list :
            if abs(res[0]-hope[0])< error_threshold and abs(res[1]-hope[1])< error_threshold :
                test = '\033[32mPASSED\033[0m'
                test_result[name]+=1
            else :
                test = '\033[31mFAILED\033[0m'
        elif type(hope)==np.ndarray :
            
            if np.all(np.add(-1*res,hope)< np.ones([len(hope),len(hope)])*error_threshold) :
                test = '\033[32mPASSED\033[0m'
                test_result[name]+=1
            else :
                test = '\033[31mFAILED\033[0m'
        else :
            
            if abs(res-hope)< error_threshold :
                test = '\033[32mPASSED\033[0m'
                test_result[name]+=1
            else :
                test = '\033[31mFAILED\033[0m'
        entrees=""

        number_of_test[name]+=1
        
        for i in range(len(entree)):
            if i == 0 :
                entrees += str(entree[i])
            else :
                entrees += ","+str(entree[i])
            
        print(6*" ","{}({})={} ; Expected result={}".format(name,entrees,res,hope))
        print(6*" ","---> test {}".format(test))
        return None

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
    print("structural_test__calc_ti : ")


    A = np.array([[1,2],[2,3]])
    T = np.zeros([2,2])
    i=0
    res = calc_ti(A,T,i)
    hope = 1
    main_test("calc_ti",[A,T,i],res,hope)

    A = np.array([[1,1,1,1],[1,5,5,5],[1,5,14,14],[1,5,14,15]])
    T = np.array([[1,0,0,0],[1,2,0,0],[1,2,0,0],[1,2,0,0]])
    i = 2
    res = calc_ti(A,T,i)
    hope = 3
    main_test("calc_ti",[A,T,i],res,hope)
    
################################################################
################################################################
################################################################
################################################################
################################################################
################################################################

    print("structural_test__calc_tji : ")

    A = np.array([[1,2],[2,3]])
    T = np.array([[1,0],[0,0]])
    i = 0
    j = 0
    res = calc_tji(A,T,i,j)
    hope = 1
    main_test("calc_tji",[A,T,i,j],res,hope)

    A = np.array([[1,1,1,1],[1,5,5,5],[1,5,14,14],[1,5,14,15]])
    T = np.array([[1,0,0,0],[1,2,0,0],[1,2,3,0],[1,2,0,0]])
    i = 2
    j = 3
    res = calc_tji(A,T,i,j)
    hope = 3
    main_test("calc_tji",[A,T,i,j],res,hope)

################################################################
################################################################
################################################################
################################################################
################################################################
################################################################

    print("structural_test__facto_cholesky_dense_REC : ")

    A = np.array([[4,-6,8,2],[-6,10,-15,-3],[8,-15,26,-1],[2,-3,-1,62]])
    T = np.zeros([4,4])
    i = 0
    res = facto_cholesky_dense_REC(A,T,i)
    hope = np.array([[2,0,0,0],[-3,1,0,0],[4,-3,1,0],[1,0,-5,6]])
    main_test("facto_cholesky_dense_REC",[A,T,i],res,hope)

    A = np.array([[1,1,1,1],[1,5,5,5],[1,5,14,14],[1,5,14,15]])
    T = np.array([[1,0,0,0],[1,2,0,0],[1,2,0,0],[1,2,0,0]])
    i = 2
    res = facto_cholesky_dense_REC(A,T,i)
    hope = np.array([[1,0,0,0],[1,2,0,0],[1,2,3,0],[1,2,3,1]])
    main_test("facto_cholesky_dense_REC",[A,T,i],res,hope)

################################################################
################################################################
################################################################
################################################################
################################################################
################################################################

    print("structural_test__facto_cholesky_dense : ")

    A = np.array([[4,-6,8,2],[-6,10,-15,-3],[8,-15,26,-1],[2,-3,-1,62]])
    res = facto_cholesky_dense(A)
    hope = np.array([[2,0,0,0],[-3,1,0,0],[4,-3,1,0],[1,0,-5,6]])
    main_test("facto_cholesky_dense",[A],res,hope)

    A = np.array([[1,1,1,1],[1,5,5,5],[1,5,14,14],[1,5,14,15]])
    res = facto_cholesky_dense(A)
    hope = np.array([[1,0,0,0],[1,2,0,0],[1,2,3,0],[1,2,3,1]])
    main_test("facto_cholesky_dense",[A],res,hope)

    A = np.array([[4,12,-16],[12,37,-43],[-16,-43,98]])
    res = facto_cholesky_dense(A)
    hope = np.array([[2,0,0],[6,1,0],[-8,5,3]])
    main_test("facto_cholesky_dense",[A],res,hope)

    A = np.array([[1,0,3],[0,4,2],[3,2,11]])
    res = facto_cholesky_dense(A)
    hope = np.array([[1,0,0],[0,2,0],[3,1,1]])
    main_test("facto_cholesky_dense",[A],res,hope)


    print_summary(function_name,test_result,number_of_test)
