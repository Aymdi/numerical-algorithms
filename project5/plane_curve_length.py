import numpy as np
import matplotlib.pyplot as plt
from cubic_spline import splint
from cubic_spline import splint_der

#approximation d'ordre 0
def rectangle(X=range(5), f=range(5)): #rectangle_droit
    """Computes the integral of the function g on the interval [0,T] using the rectangles method
    Parameters:
    ----------
    X: an array representing xi
    f: an array representing f(xi)
    """
    n = len(X)
    I = 0
    for i in range (n-1): 
        I+=(X[i+1]-X[i])*f[i]
    return I

#approximation d'ordre 1
def trapeze(X=range(5), f=range(5)):
    """Computes the integral of the function g on the interval [0,T] using the trapeze method
    Parameters:
    ----------
    X: an array representing xi
    f: an array representing f(xi)
    """
    n = len(X)
    I = 0
    for i in range(1,n):
        I += (X[i]-X[i-1])*((f[i-1]+f[i])/2)
    return I



#approximation d'ordre 2 
def simpson(X=range(5), f=range(5), n=100):
    """Computes the integral of the function g on the interval [0,T] using the simpson method
    Parameters:
    ----------
    X: an array representing xi
    f: an array representing f(xi)
    n: (b-a)/h
    """
    l = len(X)
    a = X[0]
    b = X[l-1]
    
    h = (b-a)/n
    #m = a + h/2
    #fm = splint(X, Y, m ) 
    somme = (splint(X,f,a) + splint(X, f, b))/2 + 2 * splint(X,f,a + h / 2)  # On initialise la somme
    x = a + h # La somme commence à x_1
    h_divide = h/2
    for i in range(1, n): # On calcule la somme 
        somme += splint(X,f,x) + 2 * splint(X, f, x + h_divide)
        x += h
    return somme * h / 3 # On retourne cette somme fois le pas / 3



def plane_curve_length(X, Y, T, M):
    """Computes the length of plane curves
    Parameters:
    -----------
    Y: an array : the function representing the airfoil.
    
    X: an array : points of the interval of definition of the function airfoil
    
    T: an integer : is such that f is defined and differentiable on the interval [0,T].
    
    M: a string : the method used to compute the integration.
    
    Return:
    ------
    an integer : the length of plane curves.
    """
    Y1 = []
    for i in range(len(X)):
        Y1.append(np.sqrt(splint_der(X, Y, X[i])**2 + 1))
    
    method = {"rectangle":rectangle,"trapeze":trapeze,"simpson":simpson}
    
    for key in method:
        if key == M:
            if key == "simpson" :
                return method[key](X=X,f=Y1,n=100)
            return method[key](X=X,f=Y1)
        

    return None
    
     
def test_rectangle():
    print("test on renctangle method:")
    #f(x) = 3x+1
    #the result must be 14"""
    X =[1,1.5,2,2.5,3]#the more points we add the more accurate the results are
    f =[4,5.5,7,8.5,10]
    I = rectangle(X=X,f=f)
    print("f(x)=3x+1, T=[1,3]")
    print("expexted I =14, but found I=",I)
    print("\n")
    return None
    
def test_trapeze():
    print("test on the trapeze method:")
    #f(x)= 3x+1 -> 14
    X =[1,1.5,2,2.5,3]
    f =[4,5.5,7,8.5,10]
    I = trapeze(X=X,f=f)
    print("f(x)=3x+1, T=[1,3]")
    print("expected I=14, but found I=",I)
    #f1(x) =x²+3x+2 ->38/3~=12,66
    X1=[0,1,2]
    f1=[2,6,12]
    print("f(x) =x²+3x+2, T=[0,2]")
    print("expected I~=12.66, but found I=",trapeze(X=X1, f=f1))
    #f2(x) = 5x³ + 54x² + 4x + 47 -> 266
    X2 = [0, 0.5, 1, 1.5, 2]
    f2 = [47, 63.125, 110, 191.375, 311]
    print("f(x)=5x³ + 54x² + 4x + 47, T=[0.2]")
    print("expected I = 266, but found I=",trapeze(X=X2, f=f2))
    print("\n")
    return None
    
def test_simpson():
    print("test on the Simpson method:")
    #f2(x) = 5x³ + 54x² + 4x + 47 -> 266
    X2 = [0, 0.5, 1, 1.5, 2]
    f2 = [47, 63.125, 110, 191.375, 311]
    print("f(x) = 5x³ + 54x² + 4x + 47 :\n expected I = 266, but found I =", simpson(X=X2,f=f2))
    #f3(x) = cos(x) -> 0 
    X3 = [0,np.pi]
    f3 = [1,-1]
    print("f(x) = cos(x):\nexpected I = 0, but foundI =", simpson(X=X3,f=f3))
    return None

def test_plane_curve_length():
    # f(x) = cos(x)
    X = [0, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, np.pi ]
    Y = [1, np.sqrt(2)/2, 0.5, 0, -0.5, -np.sqrt(2)/2, -1 ]
    print("\n")
    print("test on plane_curve_length:")
    print("f(x)=cos(x):")
    print("rectangle:\n",plane_curve_length(X, Y, np.pi, "rectangle" ))
    print("trapeze:\n",plane_curve_length(X, Y, np.pi, "trapeze" ))
    print("simpson:\n",plane_curve_length(X, Y, np.pi, "simpson" ))#3.81861
    #f2(x) = 1/(x²+1)
    X = [0, 1, 2, 3, 4 ]
    Y = [1, 0.5, 1/5, 1/10, 1/17 ]
    print("\nf(x)=1/(x²+1)")
    print("rectangle:\n",plane_curve_length(X, Y, 10, "rectangle" ))
    print("trapeze:\n",plane_curve_length(X, Y, 10, "trapeze" ))
    print("simpson:\n",plane_curve_length(X, Y, 10, "simpson" ))
    
if __name__ == "__main__" :

    test_rectangle()
    test_trapeze()
    test_simpson() 
    test_plane_curve_length()
    
   
    
    def convergence(method,num,f,I_th,xmin,xmax) :
        
         i = np.arange(50,num,5)
         
         def affich(num,method):
            point = np.linspace(xmin,xmax,num+1)
            image = f(point)
            if method.__name__ != "simpson" :
                return method(X=point,f=image)
            return method(X=point,f=image)
            
         I = [affich(k,method) for k in i]
         plt.plot(i-1,I,"ob",label="I(n)")
         plt.plot([0,num],[I_th,I_th],"-g",label="I_th")
         plt.xlabel("n")
         plt.ylabel("I")
         plt.grid()
         plt.legend()
         plt.show()
        
         return None
     
    f = lambda x : x**2+x**3+1000
    I_th = (1/3*(20)**3+1/4*20**4+1000*20)-(1/3*(-20)**3+1/4*(-20)**4+1000*(-20))
    xmin = -20
    xmax = 20
    convergence(rectangle,500,f,I_th,xmin,xmax)
    convergence(trapeze,500,f,I_th,xmin,xmax)
    convergence(simpson,75,f,I_th,xmin,xmax)
    
    def relative_error(method,num,f,I_th,xmin,xmax) :
        
         ex = np.linspace(xmin,xmax,500)
         i = range(50,num,5)
         
         def affich(num,method):
            point = np.linspace(xmin,xmax,num+1)
            image = f(point)
            if method.__name__ != "simpson" :
                return abs(I_th-method(X=point,f=image))/I_th
            return abs(I_th-method(X=point,f=image))/I_th
            
         I = [affich(k,method) for k in i]
         plt.plot(i,I,"ob",label="erreur relative de I(n)")
         
         if method.__name__ == "rectangle" :
             M = 0.01
             conv = [M*(ex[-1]-ex[0])**2/(2*k) for k in i]
             plt.plot(i,conv,"-g",label="M(b-a)²/(2*n) avec M={}".format(M))
         if method.__name__ == "trapeze" :
             M = 1e-4
             conv = [M*(ex[-1]-ex[0])**3/(12*k**2) for k in i]
             plt.plot(i,conv,"-g",label="M(b-a)**3/(12*n²) avec M={}".format(M))
         if method.__name__ == "simpson" :
             M = 0.0005
             conv = [M*(ex[-1]-ex[0])**5/(2880*k**4) for k in i]
             plt.plot(i,conv,"-g",label="M(b-a)**5/(2880*n**4) avec M={}".format(M))
         plt.xlabel("n")
         plt.ylabel("erreur relative")
         plt.grid()
         plt.legend()
         plt.show()
        
         return None
    
    relative_error(rectangle,500,f,I_th,xmin,xmax)
    relative_error(trapeze,500,f,I_th,xmin,xmax)
    relative_error(simpson,75,f,I_th,xmin,xmax)
            
        
    
    
