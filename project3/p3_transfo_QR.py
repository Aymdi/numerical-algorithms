import numpy as np
import matplotlib.pyplot as plt
from p2_bidiagonale import *
from tests import *


def transfo_QR(S,k):
  '''

  returns the tuple (U,S,V)

  parameters
  ----------
  S : matrix of type array
  k : iteration rank

  returns
  --------
  U : matrix of type array
  S : matrix of type array
  V : matrix of type array

  '''

  U = np.identity(S.shape[0])
  V = np.identity(S.shape[1])

  for i in range (k):
    (Q1, R1) = np.linalg.qr(S.T)
    (Q2, R2) = np.linalg.qr(R1.T)
    S = R2
    U = U.dot(Q2)
    V = np.dot(Q1.T,V)

  return (U,S,V)


def transfo_QR_verify(S, k):
  '''

  Checks the invariant BD = U*S*V at each iteration

  parameters
  ----------
  S : matrix of type array
  k : iteration rank 

  returns
  --------
  boolean

  '''
  U = np.identity(S.shape[1])
  V = np.identity(S.shape[0])
  BD = S
  for i in range (k):
    (Q1, R1) = np.linalg.qr(S.T)
    (Q2, R2) = np.linalg.qr(R1.T)
    S = R2
    U = U.dot(Q2)
    V = (Q1.T).dot(V)

    if ((BD - np.dot(U,np.dot(S,V)) >= 10**(-10)).all()):
      return False
  return True


def sum_extra_diag_terms(M):
  '''

  returns the sum of the extra diagonal elements of the matrix 
    
  Parameters
  -----------
  M : matrix of type array

  Returns
  ----------
  output : the sum

  '''
  h = M.shape[0]
  w = M.shape[1]
  sum = 0

  for i in range(h):
    for j in range(w):
      if i != j:
        sum += abs(M[i][j])

  return sum


def norm_extra_diag_terms(M):
  '''

  returns the norm of extra diagonal terms 
    
  Parameters
  -----------
  M : matrix of type array

  Returns
  ----------
  output : the norm

  '''
  d = M.diagonal(1)
  norm = np.linalg.norm(d)
  
  return norm


def plot_convergence_facto_QR(S, filename):
  y = 100
  convergence_facto_QR = []
  convergence_facto_QR.append(norm_extra_diag_terms(S))

  for i in range(y-1):
    (U, S, V) = transfo_QR(S, 1)
    convergence_facto_QR.append(norm_extra_diag_terms(S))

  fig=plt.figure()
  plt.plot(range(y), convergence_facto_QR)
  plt.xlabel("Nombre d'itérations de la factorisation QR")
  plt.ylabel("Norme de l'extra-diagonale")
  plt.title("Convergence vers une matrice diagonale")
  plt.savefig("img/"+filename+".png", dpi=fig.dpi)


#Q3
'''

Montrons l'invariant suivant: "S, R1 et R2 sont toujours bidiagonales".

À l'entrée du programme, la matrice S est déjà bidiagonale supérieure.
t(S) est bidiagonale inférieure.
Comme l'algorithme QR préserve la forme de la matrice tridiagonale à chaque étape (slide57 chp3),
alors R1 est une matrice tridiagonale et comme elle est aussi triangulaire supérieure
on en déduit que R1 est bidiagonale supérieure. On suit le meme raisonnement pour prouver
que R2 est aussi bidiagoanle supérieure.

'''


def transfo_QR_simplify(S,k):
  '''

  returns the tuple (U,S,V)

  parameters
  ----------
  S : matrix of type array
  k : iteration rank

  returns
  --------
  U : matrix of type array
  S : matrix of type array
  V : matrix of type array

  '''
  U = np.identity(S.shape[1])
  V = np.identity(S.shape[0])

  for i in range (k):
    (Q,R) = np.linalg.qr(S)
    S = R.dot(Q)

  return U,S,V


def modifyUS(U, S):
  '''

  Modifies the U and S matrices in order to preserve the U * S product and the properties of S.

  parameters
  ----------
  U : matrix of type array
  S : matrix of type array

  returns
  --------
  U : matrix of type array
  S : matrix of type array

  '''
  n = S.shape[0]

  D1 = abs(np.diag(S)).tolist()
  D2 = np.diag(S).tolist()

  T1 = D1.copy()
  T1.sort(key=abs,reverse=True)
  T2 = D2.copy()
  T2.sort(key=abs,reverse=True)

  for i in range(n):
    S[i][i] = T1[i]
    U[:,i] = U[:,i] * D1[i] / T2[i]
    if (T2[i]<0) :
      for k in range(n) :
        U[k][i] *= -1
  return U,S

def test__modifyUS(U, S):
  R1 = np.dot(U,S)
  (U,S) = modifyUS(U,S)
  R2 = np.dot(U,S)
  if ( (R1==R2).all() ):
    return 0
  return 1


def p3_tests():

    print("MODULE : PARTIE_3")
    function_name = ["transfo_QR","transfo_QR_verify","sum_extra_diag_terms","norm_extra_diag_terms","transfo_QR_simplify","modifyUS"]
    (test_result, number_of_test) = init_tests(function_name)

    ###################################
    U = np.array([[1,1,1],[1,1,1],[1,1,1]])
    S = np.array([[2,0,0],[0,1,0],[0,0,3]])
    res = test__modifyUS(U,S)
    hope = 1
    main_test("modifyUS",[U,S],res,hope, test_result, number_of_test)

    

    # (U,S,V) = transfo_QR(BD,10)
    # print(U)
    # print(S)
    # print(V)

    # print(count_extra_diag_terms(S))
    # print(transfo_QR_verify(BD,10))
    # plot_convergence_facto_QR(BD)


    img_full = np.array(plt.imread("img/essai.png"))

    A1 = take_nth_pixel(img_full, 0)
    (QL1, BD1, QR1) = bidiagonalize(A1)

    A2 = take_nth_pixel(img_full, 1)
    (QL2, BD2, QR2) = bidiagonalize(A2)

    A3 = take_nth_pixel(img_full, 2)
    (QL3, BD3, QR3) = bidiagonalize(A3)

    plot_convergence_facto_QR(BD1, "convergence_A1")
    # plot_convergence_facto_QR(BD2, "convergence_A2")
    # plot_convergence_facto_QR(BD3, "convergence_A3")

    # print(U)
    # print(S)
    # print(V)

    ###################################

    print_summary(function_name,test_result,number_of_test)


if __name__=="__main__" :

    p3_tests()











