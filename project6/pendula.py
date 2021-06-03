import matplotlib.pyplot as plt
import numpy as np
from diff_equation_resol import *
import matplotlib.animation as animation
from random import randint
from math import sin
from math import cos



G = 9.81  # acceleration due to gravity
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg








def pendula(y,t):
    #approached solution for a samll theta

    return np.array([y[1],-(G/L1)*y[0]])


def doublePendula(y,t):
    th1 = y[0]
    thd1 = y[1]
    th2 = y[2]
    thd2 = y[3]
    #print((L2*((2*M1+M2) - M2*2*cos(th1 - th2)*sin(th1 - th2))))
    yp1 = (-(2*M1 +M2)*G*sin(th1) - G*sin(th1 - 2*th2)*M2 - 2*sin(th1 - th2)*M2*((thd2**2)*L2 + (thd1**2)*L1*cos(th1 - th2)))/(L1*((2*M1+M2) - cos(2*th1 - 2*th2)*M2))

    yp2 = (2 * sin(th1 - th2)*(thd1**2*L1*(M1+M2) + (M1+M2)*G*cos(th1) + thd2**2*L2*M2*cos(th1 - th2)))/(L2*((2*M1+M2) - M2*cos(2*th1 - 2*th2)))

    return np.array([thd1,yp1,thd2, yp2])




def resolve_pendula(thetaInit):
    """
    thetha init in rad
    resolvs the pendula equation
    """
    PbCauchy2 = ([np.array([thetaInit,0]),0],pendula)
    y_euler_dim2, t_euler_dim2 = meth_epsilon(PbCauchy2[0][0], PbCauchy2[0][1], 5, 0.01, PbCauchy2[1], step_Runge_Kutta)
    return (y_euler_dim2, t_euler_dim2)

def resolveDoublePendula(thetaInit,thetaInit2):
    """
    thetha init in rad
    resolvs the pendula equation
    """
    PbCauchy2 = ([np.array([thetaInit,0,thetaInit2,0]),0],doublePendula)
    y_euler_dim2, t_euler_dim2 = meth_epsilon(PbCauchy2[0][0], PbCauchy2[0][1], 5, 0.01, PbCauchy2[1], step_Runge_Kutta)
    return (y_euler_dim2, t_euler_dim2)

def drawPendulaTime(thetaInit):
    """
    thetaInit in rad
    draws the trajectory x and y of a pendula
    """
    theta, t= resolve_pendula(thetaInit)
    # print("wes")
    theta = np.array(theta)[:,0]
    #print(y.shape)
    plt.plot(t,theta)#seule le le premier résulat nous est utile


def drawPendulaTrajetory(thetaInit):
    """
    thetaInit in rad
    draws the trajectory x and y of a pendula
    """
    y,t = resolve_pendula(thetaInit)
    # print("wes")
    y = np.array(y)
    #print(y.shape)
    xb = np.sin(y[:,0])*L1
    yb = -np.cos(y[:,0])*L1
    plt.plot(xb,yb)#seule le le premier résulat nous est utile

def drawDoublePendulaTrajetory(thetaInit1,thetaInit2):
    """
    thetaInit in rad
    draws the trajectory x and y of a pendula
    """
    y,t = resolveDoublePendula(thetaInit1,thetaInit2)
    # print("wes")
    y = np.array(y)
    #print(y.shape)
    xa = np.sin(y[:,0])*L1
    ya = -np.cos(y[:,0])*L1
    xb = np.array([])
    yb = np.array([])
    for i in range(len(xa)):
        xb = np.append(xb,np.array([xa[i]+np.sin(y[i][2]) *L2]))
        yb = np.append(yb,np.array([ya[i]-np.cos(y[i][2]) *L2]))

    print(xb.shape)
    #xb = np.sin(y[:,2])
    # yb = -np.cos(y[:,2])

    plt.plot(xa,ya)#seule le le premier résulat nous est utile
    plt.plot(xb,yb)#seule le le premier résulat nous est utile

def frequencyCalculator(y,t):
    """This function takes as an input y the result of a function
    and t the time associcated
    and returns the frequency with precision epsilon
    which is 1/the time it take for 10 values to repeat
    10 was taken arbitrairly
    """
    epsilon = 0.001
    ctr = 0
    #record first sample
    # sample = randint(1,len(y) - 10)
    sample = randint(0,len(y)//2)
    timeSample = 0
    extract = y[sample]
    for i in range(0,len(y)):
        if (y[i-1] < y[i] >y[i+1]):#augmentation then stops, their has to be one in a preidoical phenomenun
            if(timeSample == 0 ):
                timeSample = t[i]
            else:
                return 1/(t[i]-timeSample)
    print("no period found")
    return None

def drawFrequencyNextToSqrt():
    a = np.array([])
    x = np.linspace(-math.pi/2,math.pi/2,10)
    for i in x:
        y,t = resolve_pendula(i)
        y = np.array(y)
        t = np.array(t)
        print(frequencyCalculator(y[:,0],t))
        a =np.append(a,frequencyCalculator(y[:,0],t))

    plt.plot(x,a)
    plt.plot(0,math.sqrt(G/L1)/(math.pi*2),'ro',c='red')#we add a point representing the expected value













if __name__ == "__main__":
    print('test')
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0,10 ), ylim=(-2, 2))
    ax.set_aspect('equal')
    plt.xlabel("Theta")
    plt.ylabel("Time")
    drawPendulaTime(math.pi/4)

    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    drawPendulaTrajetory(math.pi/4)

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
    ax.set_aspect('equal')
    plt.xlabel("Initial Theta")
    plt.ylabel("Frequency")

    drawFrequencyNextToSqrt()

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    plt.xlabel("x")
    plt.ylabel("y")


    for i in np.arange(math.pi/4,math.pi/4 +0.1,0.01):
        drawDoublePendulaTrajetory(math.pi,i)






    plt.show()
