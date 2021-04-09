# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
import numpy as np
from partie3_q1 import genere_matrice_N
sys.path.append("../partie_1")
from partie1_q4 import facto_cholesky_incomplete
sys.path.append("../partie_2")
from partie2 import solve

def probleme_mur_chaud(N,T_mur_chaud) :

    '''
    probleme_mur_chaud(...)

    returns the coefficients of the matrix T of the problem of heat diffusion in a 2D plan of size N by N with a warm wall in north position. The temperature of the warm wall is T_mur_chaud.

    Parameters
    ----------

    N : size of the 2D plan

    T_mur_chaud : the temperature of the warm wall

    Returns
    -------

    out : the matrix T of heat diffusion

    '''

    A,b = genere_matrice_N(N)
    b[:N,0] = -T_mur_chaud
    A_oppose = -1 * A
    b_oppose = -1 * b
    L = facto_cholesky_incomplete(A_oppose)
    x = solve(L,np.transpose(L),b_oppose)
    
    return x.reshape([N,N])

def probleme_radiateur_centre(N,T_radiateur) :

    '''
    probleme_radiateur_centre(...)

    returns the coefficients of the matrix T of the problem of heat diffusion in a 2D plan of size N by N with a radiator in the center. The temperature of the radiator is T_radiateur.

    Parameters
    ----------

    N : size of the 2D plan

    T_radiateur : the temperature of the radiator

    Returns
    -------

    out : the matrix T of heat diffusion result

    '''

    A,b = genere_matrice_N(N)
    b[N//2*N+N//2,0] = -T_radiateur
    A_oppose = -1 * A
    b_oppose = -1 * b
    L = facto_cholesky_incomplete(A_oppose)
    x = solve(L,np.transpose(L),b_oppose)
    
    return x.reshape([N,N])
    
def affiche_image(T) :

    '''
    affiche_image(...)

    plot the temperature color map from the heat diffusion result matrix

    Parameters
    ----------

    T : the heat diffusion result matrix

    Returns
    -------

    out : None

    '''

    plt.imshow(T,cmap=plt.cm.hot,interpolation='bilinear',aspect='equal',extent=(0,1,0,1))
    plt.colorbar()
    plt.show()

    return None


if __name__ == "__main__" :

    function_name = ["probleme_mur_chaud","probleme_radiateur_centre"]
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

    print("structural_test__probleme_mur_chaud : ")

    N = 11
    Temp = 50
    f = probleme_mur_chaud
    T = f(N,Temp)
    
    if np.all(T < np.ones([len(T),len(T)])*Temp) and np.all(T[N//2,:] < T[0,:]) and np.all(T[N-1,:] < T[N//2,:]) : # check if all Temperature are less than T_init and temperature bottom line is cooler than center line and center line is cooler than top line (warm_wall)
        test_result[f.__name__]+=1
    number_of_test[f.__name__]+=1

    N = 6
    Temp = 200
    f = probleme_mur_chaud
    T = f(N,Temp)
    
    if np.all(T < np.ones([len(T),len(T)])*Temp) and np.all(T[N//2,:] < T[0,:]) and np.all(T[N-1,:] < T[N//2,:]) : # check if all Temperature are less than T_init and temperature bottom line is cooler than center line and center line is cooler than top line (warm_wall)
        test_result[f.__name__]+=1
    number_of_test[f.__name__]+=1

    
    
    print("structural_test__probleme_radiateur_centre : ")

    N = 11
    Temp = 50
    f = probleme_radiateur_centre
    T = f(N,Temp)
    
    if np.all(T < np.ones([len(T),len(T)])*Temp) and np.all(T[0,:] < np.ones([len(T)])*T[N//2,N//2]) and np.all(T[N-1,:] < np.ones([len(T)])*T[N//2,N//2]) and np.all(T[:,0] < np.ones([len(T)])*T[N//2,N//2]) and np.all(T[:,N-1] < np.ones([len(T)])*T[N//2,N//2]) : # check if all Temperature < T_init and borders are cooler than the center
        test_result[f.__name__]+=1
    number_of_test[f.__name__]+=1

    N = 6
    Temp = 200
    f = probleme_radiateur_centre
    T = f(N,Temp)

    if np.all(T < np.ones([len(T),len(T)])*Temp) and np.all(T[0,:] < np.ones([len(T)])*T[N//2,N//2]) and np.all(T[N-1,:] < np.ones([len(T)])*T[N//2,N//2]) and np.all(T[:,0] < np.ones([len(T)])*T[N//2,N//2]) and np.all(T[:,N-1] < np.ones([len(T)])*T[N//2,N//2]) : # check if all Temperature < T_init and borders are cooler than the center
        test_result[f.__name__]+=1
    number_of_test[f.__name__]+=1

    print_summary(function_name,test_result,number_of_test)
    

