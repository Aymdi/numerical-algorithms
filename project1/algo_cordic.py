from math import log
from math import atan
import math
import numpy as np
import matplotlib.pyplot as plt

L = [log(2), log(1.1), log(1.01), log(1.001), log(1.0001), log(1.00001), log(1.000001)]
A = [atan(1), atan(0.1), atan(0.01), atan(0.001), atan(0.0001),atan(0.00001), atan(1.000001)]
'''
ETAPE1: Ramenement de la valeur à évaluer dans l'intervalle où la méthode est applicable
ETAPE2: Application de l'algorithme de CORDIC
'''
def log_cordic(x):
    '''
    Calcul du logarithme népérien par l'algorithme de CORDIC
    L'erreur commise est inférieure à 10**(-12)
    '''
    #ETAPE1
    n = 0
    while x >= 10:                    #ln(x)= ln(x*10^(-n))+n*ln(10)
        x /= 10
        n += 1
    #ETAPE2
    k, y, p = 0,0,1 
    while(k <= 6):
        while(x >= p + p*(10**(-k))):
            y = y + L[k]               #L[k] = log(1 + 10**(-k))
            p = p + p*(10**(-k))
        k = k +1
    return (y + x/p - 1) + n*log(10)   #Reponse exacte: y + log(x/p)


def exp_cordic(x):
    '''
    Calcul de l'exponetielle par l'algorithme de CORDIC
    L'erreur commise est inférieure à 10**(-12)
    '''
    #ETAPE1
    n = 0
    while x >= log(10):              #exp(x)= exp(x-n*ln(10))*10^n
        x -= log(10)
        n += 1
    #ETAPE2
    k = 0
    y = 1
    while(k <= 6):
        while(x >= L[k]):
            x = x - L[k]
            y = y + y*(10**(-k))     
        k = k + 1
    return (y + y*x) * (10**n)        #Reponse exacte: y*exp(x)


def atan_cordic(x):
    '''
    Calcul de l'arctangente par l'algorithme de CORDIC
    L'erreur commise est inférieure à 10**(-12)
    '''
    #ETAPE1
    n = 1
    m = 0
    l = 1
    if x<0:
        x = -x         #arctan(x) = -arctan(-x)
        n = -1
    if x>1:
        x = 1/x        #arctan(x) = Pi/2 - arctan(1/x)
        m = 1
        l = -1
    #ETAPE2
    k = 0
    y = 1
    r = 0
    while(k <= 4):
        while( x < y*(10**(-k))):
            k += 1
        if k <= 4: 
            xp = x - y*(10**(-k))
            y  = y + x*(10**(-k))
            x  = xp                        
            r  = r + A[k]                      #A[k]= atan(10**(-k))          
    return n*(m*np.pi/2 + l*(r + x/y))         #Reponse exacte: r + atan(x/y)


def tan_cordic(x):
    '''
    Calcul de la tangente par l'algorithme de CORDIC
    L'erreur commise est inférieure à 10**(-12)
    '''
    #ETAPE1
    x %= np.pi           #tan(x) = tan(x modulo Pi)
    l = 1
    m = 1
    if x > np.pi/2:         #si x>Pi/2 tan(x)= -tan(Pi-x)
        x = np.pi - x
        l = -1
    if x > np.pi/4:         #si x>Pi/4 tan(x)= 1/tan(Pi/2-x)
        x = np.pi/2 - x
        m = -1             
    #ETAPE2
    k = 0
    n = 0
    d = 1
    while(k <= 4):
        while(x >= A[k]):
            x = x - A[k]
            np1= n + d*(10**(-k))
            d = d - n*(10**(-k))
            n = np1
        k = k +1
    return l*(((n + x*d)/(d-x*n))**m)        #Reponse exacte: (n+d*tan(x))/(d-n*tan(x))

