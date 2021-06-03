# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:06:51 2021

@author: leogu
"""

import matplotlib.pyplot as plt
import numpy as np
from cubic_spline import get_foil
from plane_curve_length import plane_curve_length

def display_foil(file="foil.txt") :
    
    '''
    
    display_foil(...)

    diplays the foil
    
    Parameters
    ----------
    
    file : foil data file

    Returns
    -------

    out : None

    '''
    
    x_sup,y_sup,x_inf,y_inf = get_foil(0,1,500,file)
            
    
    plt.plot(x_sup,y_sup,"-b")
    plt.plot(x_inf,y_inf,"-b")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.axis('image')
    plt.show()
    
    return None

def display_airflow(file="foil.txt",airflow_line_number=10) :
    
    '''
    
    display_airflow(...)

    diplays the foil and airflow

    Parameters
    ----------
    
    file : foil data file
    
    airflow_line_number : number of airflow line for top and also for bottom

    Returns
    -------

    out : None

    '''
    
    x_sup,y_sup,x_inf,y_inf = get_foil(0,1,500,file)
    h_min = min(y_inf)
    h_max = max(y_sup)
    
    airflow_sup = lambda t,i : (1-i)*t+i*3*h_max
    airflow_inf = lambda t,i : (1-i)*t+i*3*h_min
    
    lam = np.linspace(0,1,airflow_line_number+1)
    
    for i in lam :
        
        y_airflow_sup = []
        y_airflow_inf = []
        
        for j in range(len(y_inf)) :
            y_airflow_sup.append(airflow_sup(y_sup[j],i))
            y_airflow_inf.append(airflow_inf(y_inf[j],i))
        plt.plot(x_sup,y_airflow_sup,"-k")
        plt.plot(x_inf,y_airflow_inf,"-k")
            
    
    plt.plot(x_sup,y_sup,"-b")
    plt.plot(x_inf,y_inf,"-b")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.axis('image')
    plt.show()
    
    return None


def display_pressure_map(file="foil.txt",airflow_line_number=50,cruising_speed=250,airfoil_rope=3.5,point_number=100) :
    
    '''
    
    display_pressure_map(...)

    diplays the foil and airflow

    Parameters
    ----------
    
    file : foil data file
    
    airflow_line_number : number of airflow line for top and also for bottom
    
    cruising_speed : cruising speed (m/s)
    
    airfoil_rop : airfoil rope (m)
    
    time : time of air travel for a length of 1
    
    point_number : number of computed points

    Returns
    -------

    out : None

    '''
    
    time = airfoil_rope/cruising_speed
    
    x_sup,y_sup,x_inf,y_inf = get_foil(0,1,point_number,file)
    
    h_min = min(y_inf)
    h_max = max(y_sup)
    min_f = 3*h_min
    dist = 3*h_max-3*h_min
    
    airflow_sup = lambda t,i : (1-i)*t+i*3*h_max
    airflow_inf = lambda t,i : (1-i)*t+i*3*h_min
    
    
    lam = np.linspace(0,1,airflow_line_number+1)
    
    longueur_sup = []
    longueur_inf = []
    y_airflow_sup = []
    y_airflow_inf = []
    
    for index in lam :
        
        y_computed_sup = airflow_sup(np.array(y_sup),index)
        y_computed_inf = airflow_inf(np.array(y_inf),index)
        length_sup = plane_curve_length(x_sup,y_computed_sup,1,"simpson")
        length_inf = plane_curve_length(x_inf,y_computed_inf,1,"simpson")
        y_airflow_sup.append(list(y_computed_sup))
        y_airflow_inf.append(list(y_computed_inf))
        longueur_sup.append(length_sup)
        longueur_inf.append(length_inf)
        
    y_airflow_sup = np.array(y_airflow_sup)
    y_airflow_inf = np.array(y_airflow_inf)
    
    airflow = np.concatenate((y_airflow_sup,y_airflow_inf))
    airflow = np.transpose(airflow)
    longueur = longueur_sup+longueur_inf
    
    
    A = np.full([point_number,point_number],0,dtype = float)
    
    recadre_j = dist/point_number
    alpha = 1/2*(1/time)**2
    size = point_number-1
    
    for i in range(point_number) :
        
        air = airflow[i]
        recadre_i = min_f-air
        
        
        for j in range(point_number) :
            
            norm = abs(j*recadre_j+recadre_i)
                
            long = longueur[norm.argmin()-1]
                
            A[size-j][i] = alpha*(long*airfoil_rope)**2
            
    
    plt.title(label = "airfoil rope = {}m, cruising speed = {}m/s".format(airfoil_rope,cruising_speed))   
    plt.xlabel("x")
    plt.ylabel("z")
    
    
    plt.imshow(A,cmap="hot",extent=(0,1,3*h_min,3*h_max))
    color = plt.colorbar()
    color.set_label("Pa")
    plt.show()
    
    return None

def display_reynolds(speed=900*10**3/3600) :
    
    '''
    
    display_reynolds(...)

    diplays the air reynolds according to the caracteristic length

    Parameters
    ----------
    
    speed : airflow speed
    
    Returns
    -------

    out : None

    '''
    
    
    cynematic_viscosity = 15.6*10**(-6)
    
    R = speed/cynematic_viscosity
    
    for i in range(5) :
        
        for j in [1,2,4,6,8] :
            
            L = j*10**(i-6)
            if i == 0 and j==1 :
                plt.plot(L,R*L,"b.",label="Nombre de Reynolds de l'air")
            else :
                plt.plot(L,R*L,"b.")
           
    plt.plot([0,10**-1],[2000,2000],"-g",label="Nombre de Reynolds max pour un écoulement laminaire")
    plt.plot([0,10**-1],[3000,3000],"-r",label="Nombre de Reynolds min pour un écoulement turbulent")
    plt.xlabel("Longueur caractéristique")
    plt.ylabel("Nombre de Reynolds")
    plt.xscale("log")  
    plt.yscale("log") 
    plt.title(label="vitesse ={}m/s".format(speed))
    plt.grid()
    plt.legend()  
    plt.show()
    return None
        
       
