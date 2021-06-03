# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 11:27:08 2021

@author: leogu
"""

import numpy as np
import matplotlib.pyplot as plt
from cubic_spline import splint

def display_rectangle(f,xmin,xmax,n) :
    
    '''
    
    display_rectangle(...)

    draws the right rectangle method for function f between xmin and xmax for n-intermediate points

    Parameters
    ----------
    
    f : function
    
    xmin : the minimum x axis
    
    xmax : the maximum x axis
    
    n : number of intermediate points
    
    Returns
    -------

    out : None

    '''
    
    
    x = np.linspace(xmin,xmax,500)
    alpha_i = np.linspace(xmin,xmax,n)
    y = f(x)
    
    for i in range(len(alpha_i)-1) :
        
        plt.plot([alpha_i[i],alpha_i[i]],[0,f(alpha_i[i])],'-b')
        plt.plot([alpha_i[i+1],alpha_i[i+1]],[0,f(alpha_i[i])],'-b')
        plt.plot([alpha_i[i],alpha_i[i+1]],[f(alpha_i[i]),f(alpha_i[i])],'-b')
    
    plt.plot(x,y,"-r",label="f")
    plt.plot([xmin,xmax],[0,0],"-k")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.show()
    
    return None

def display_trapeze(f,xmin,xmax,n) :
    
    '''
    
    display_trapeze(...)

    draws the trapeze method for function f between xmin and xmax for n-intermediate points

    Parameters
    ----------
    
    f : function
    
    xmin : the minimum x axis
    
    xmax : the maximum x axis
    
    n : number of intermediate points
    
    Returns
    -------

    out : None

    '''
    
    x = np.linspace(xmin,xmax,500)
    alpha_i = np.linspace(xmin,xmax,n)
    y = f(x)
    
    for i in range(1,len(alpha_i)) :
        
        plt.plot([alpha_i[i-1],alpha_i[i-1]],[0,f(alpha_i[i-1])],'-b')
        plt.plot([alpha_i[i],alpha_i[i]],[0,f(alpha_i[i])],'-b')
        plt.plot([alpha_i[i-1],alpha_i[i]],[f(alpha_i[i-1]),f(alpha_i[i])],'-b')
    
    plt.plot(x,y,"-r",label="f")
    plt.plot([xmin,xmax],[0,0],"-k")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.show()
    
    return None

def display_simpson(f,xmin,xmax,n) :
    
    '''
    
    display_simpson(...)

    draws the simpson method for function f between xmin and xmax for n-intermediate points

    Parameters
    ----------
    
    f : function
    
    xmin : the minimum x axis
    
    xmax : the maximum x axis
    
    n : number of intermediate points
    
    Returns
    -------

    out : None

    '''
    
    x = np.linspace(xmin,xmax,500)
    alpha_i = np.linspace(xmin,xmax,n)
    y = f(x)
    
    for i in range(len(alpha_i)-1) :
        
        m = (alpha_i[i]+alpha_i[i+1])/2
        X = [alpha_i[i],m,alpha_i[i+1]]
        fi = f(alpha_i[i])
        fj = f(alpha_i[i+1])
        Y = [fi,f(m),fj]
        
        x_i = np.linspace(alpha_i[i],alpha_i[i+1],20)
        y_i = []
        
        for alpha in x_i :
            
            y_i.append(splint(X,Y,alpha))
            
        plt.plot([alpha_i[i],alpha_i[i]],[0,fi],'-b')
        plt.plot([alpha_i[i+1],alpha_i[i+1]],[0,fj],'-b')
        plt.plot(x_i,y_i,'-b')
    
    plt.plot(x,y,"-r",label="f")
    plt.plot([xmin,xmax],[0,0],"-k")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.show()
    
    return None

if __name__ == "__main__" :
    
    f = lambda x : x**3+x**2+1000
    display_rectangle(f,-20,20,20)
    display_trapeze(f,-20,20,5)
    display_simpson(f,-20,20,3)