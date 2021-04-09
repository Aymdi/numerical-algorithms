import numpy as np
import copy
from partie1_q1 import facto_cholesky_dense
from partie1_q3 import generate_SPDSM
from partie1_q4 import facto_cholesky_incomplete


def calc_inverse_matrice_T(T) :

    res = copy.deepcopy(T)
    for i in range(len(res)):
        for j in range(i) :
            res[i,j] = -res[i,j]/(res[j,j]*res[i,i])
    diag = np.diag(res)
    
    res += np.diag(1/diag) - np.diag(diag)

    return res
'''
A = generate_SPDSM(5,5)
print("A = ",A)
L = facto_cholesky_dense(A)
LL = np.dot(L,np.transpose(L))
LT = np.transpose(L)
print(LT)
LT2 = calc_inverse_matrice_T(LT)
print(LT2)
#print(LT2-np.linalg.inv(LT))
LL2 = np.dot(calc_inverse_matrice_T(np.transpose(L)),calc_inverse_matrice_T(L))
#print(np.dot(LL,LL2))
'''

def est_bon_preconditionneur(T, A) :
    '''
    est_bon_preconditionneur(...)

    checks if T is a good preconditioner of A 

    Parameters
    ----------

    A : positive definite square matrix

    T : the preconditioner

    Returns
    -------

    out : boolean telling if T is a good preconditioner of A 

    '''

    return np.linalg.cond( np.transpose(np.linalg.inv(T)) * np.linalg.inv(T) * A, np.inf ) <= np.linalg.cond( A, np.inf )


def test_est_bon_preconditionneur() :
    A = generate_SPDSM(18, 18)
    T_dense = facto_cholesky_dense(A)
    T_incomplete = facto_cholesky_incomplete(A)

    print("A =",A)

    print("cond(A) = ", np.linalg.cond(A, np.inf))
    #T_dense
    print("cond(T_dense) = ", np.linalg.cond( np.transpose(np.linalg.inv(T_dense)) * np.linalg.inv(T_dense) * A, np.inf ))
    print("est_bon_preconditionneur(T_dense,A)",est_bon_preconditionneur(T_dense,A),sep=" = ")
    #T_incomplete
    print("cond(T_incomplète) = ", np.linalg.cond( np.transpose(np.linalg.inv(T_incomplete)) * np.linalg.inv(T_incomplete) * A, np.inf ))
    print("est_bon_preconditionneur(T_incomplete, A)",est_bon_preconditionneur(T_incomplete, A),sep=" = ")

if __name__=="__main__" :
    
    test_est_bon_preconditionneur()
