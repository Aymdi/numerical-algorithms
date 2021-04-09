import numpy as np

def genere_matrice_N(N) :

    '''
    genere_matrice_N(...)

    returns the matrix A and b of the linear system of the 2D heat diffusion problem on a N by N plan

    Parameters
    ----------

    N : positive integer which is the size of the plan

    Returns
    -------

    out : a tuple with the matrix A as first element and the vector b as second element

    Space complexity
    ----------------

    O(N**4)

    '''

    if N <= 0 :
        return None

    A = np.diag([-4 for k in range(N**2)])+np.diag([int((k+1)%N!=0) for k in range(N**2-1)],1)+np.diag([int((k+1)%N!=0) for k in range(N**2-1)],-1)+np.diag([1 for k in range(N**2-N)],-N)+np.diag([1 for k in range(N**2-N)],N)
    return (((N+1)**2) *A,np.zeros([N**2,1]))

if __name__ == "__main__" :

    function_name = ["genere_matrice_N"]
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
    print("structural_test__genere_matrice_N : ")
    error_threshold = 10**(-6)
    
    N = 1
    f = genere_matrice_N
    A,b = f(N)

    if abs(A[0,0]-(-4*(N+1)**2)) < error_threshold and len(A) == N**2 and np.count_nonzero(b) == 0 and len(b) == N**2 :
        test_result[f.__name__]+=1
    number_of_test[f.__name__]+=1

    N = 5
    f = genere_matrice_N
    A,b = f(N)
    hope = np.diag([-4 for k in range(N**2)])+np.diag([int((k+1)%N!=0) for k in range(N**2-1)],1)+np.diag([int((k+1)%N!=0) for k in range(N**2-1)],-1)+np.diag([1 for k in range(N**2-N)],-N)+np.diag([1 for k in range(N**2-N)],N)
    hope = (N+1)**2*hope
    
    
    if np.all(np.add(-1*hope,A)< np.ones([len(hope),len(hope)])*error_threshold) and np.count_nonzero(b) == 0 and len(b)==N**2 :
        test_result[f.__name__]+=1
    number_of_test[f.__name__]+=1

    print_summary(function_name,test_result,number_of_test)
