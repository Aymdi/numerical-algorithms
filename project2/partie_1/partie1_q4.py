import matplotlib.pyplot as plt
import numpy as np
import time
from partie1_q1 import calc_ti,calc_tji,facto_cholesky_dense
from partie1_q3 import generate_SPDSM

def facto_cholesky_incomplete_REC(A,T,i):

    '''
    facto_cholesky_incomplete_REC(...)

    completes T with the value of tii and tji for j from 0 to i-1 of the incomplete Cholesky factorization and returns a call of itself for next index

    Parameters
    ----------

    A : positive definite square matrix

    T : the result of incomplete Cholesky factorization

    i : positive integer between 0 and the dimension of A-1

    Returns
    -------

    out : itself with a call on next index

    '''

    if i == len(A) :
        return T
    T[i,i]=calc_ti(A,T,i)
        
    for j in range(i+1,len(A)) :
        if A[j,i] != 0 :    
            T[j,i] = calc_tji(A,T,i,j)
    return facto_cholesky_incomplete_REC(A,T,i+1)
    
def facto_cholesky_incomplete(A) :

    '''
    facto_cholesky_incomplete(...)

    returns the matrix of the incomplete Cholesky factorization 

    Parameters
    ----------

    A : positive definite square matrix

    Returns
    -------

    out : the matrix of the incomplete Cholesky factorization

    Time complexity
    ---------------

    O(2*i**2)

    Space complexity
    ----------------

    O(n**2)

    '''

    T = np.zeros([len(A),len(A)])
    return facto_cholesky_incomplete_REC(A,T,0)

def compare_function(f1,f2):

    '''
    compare_function(...)

    print a graph with the curve of time execution of function f1 and f2 applied to a symetrical positive definite sparse matrix 

    Parameters
    ----------

    f1 : function 1 which can work on an symetrical positive definite sparse matrix

    f2 : function 2 which can work on an symetrical positive definite sparse matrix

    Returns
    -------

    out : None

    '''

    timer_f1=[]
    timer_f2=[]
    taille = 20
    nb_matrice = 20
    test_sur_matrice = 10

    for i in range(8,taille) :
        timer_1_taille_i =[]
        timer_2_taille_i =[]
        for j in range(nb_matrice) :
            A = generate_SPDSM(i,i)
            timer = time.time()
            for k in range(test_sur_matrice) :
                f1(A)
            timer_1 = (time.time()-timer)/test_sur_matrice
            timer = time.time()
            for k in range(test_sur_matrice) :
                f2(A)
            timer_2 = (time.time()-timer)/test_sur_matrice
            timer_1_taille_i.append(timer_1)
            timer_2_taille_i.append(timer_2)
        timer_f1.append(sum(timer_1_taille_i)/len(timer_1_taille_i))
        timer_f2.append(sum(timer_2_taille_i)/len(timer_2_taille_i))
    
    plt.plot(range(8,taille),timer_f1,label=f1.__name__)
    plt.plot(range(8,taille),timer_f2,label=f2.__name__)
    plt.grid()
    plt.xlabel("taille de la matrice")
    plt.ylabel("temps en secondes")
    plt.legend()
    plt.savefig("Graphe.png")
    plt.show()

    return None

if __name__=="__main__" :

    function_name = ["facto_cholesky_incomplete"]
    test_result = {}
    number_of_test = {}

    for name in function_name :
        test_result[name]=0
        number_of_test[name]=0
        
    def main_test(name,entree,res,hope):

        error_threshold = 0.000001
        
        if np.all(np.add(-1*res[0],hope[0])< np.ones([len(hope[0]),len(hope[0])])*error_threshold) and hope[1] == res[1]:
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
    print("structural_test__facto_cholesky_incomplete_REC : ")

    print("structural_test__facto_cholesky_dense : ")
    
    A = generate_SPDSM(5,2)
    T = np.zeros([4,4])
    i = 0
    timer = time.time()
    nb_timer = 10
    for i in range(nb_timer):
        facto_cholesky_incomplete(A)
    timer_in = (time.time()-timer)/nb_timer
    for i in range(nb_timer):
        facto_cholesky_dense(A)
    timer_de = time.time()
    timer = ((timer_de-timer_in)/nb_timer) > timer_in
    res = [facto_cholesky_incomplete(A),timer]
    hope = [facto_cholesky_dense(A),True]
    
    main_test("facto_cholesky_incomplete",[A],res,hope)

    A = generate_SPDSM(8,2)
    T = np.zeros([4,4])
    i = 0
    timer = time.time()
    nb_timer = 10
    for i in range(nb_timer):
        facto_cholesky_incomplete(A)
    timer_in = (time.time()-timer)/nb_timer
    for i in range(nb_timer):
        facto_cholesky_dense(A)
    timer_de = time.time()
    timer = ((timer_de-timer_in)/nb_timer) > timer_in
    res = [facto_cholesky_incomplete(A),timer]
    hope = [facto_cholesky_dense(A),True]
    
    main_test("facto_cholesky_incomplete",[A],res,hope)

    A = generate_SPDSM(15,5)
    T = np.zeros([4,4])
    i = 0
    timer = time.time()
    nb_timer = 10
    for i in range(nb_timer):
        facto_cholesky_incomplete(A)
    timer_in = (time.time()-timer)/nb_timer
    for i in range(nb_timer):
        facto_cholesky_dense(A)
    timer_de = time.time()
    timer = ((timer_de-timer_in)/nb_timer) > timer_in
    res = [facto_cholesky_incomplete(A),timer]
    hope = [facto_cholesky_dense(A),True]
    
    main_test("facto_cholesky_incomplete",[A],res,hope)

    print_summary(function_name,test_result,number_of_test)
