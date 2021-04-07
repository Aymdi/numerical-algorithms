from math import * #import tous les modules de maths donc pas besoin de faire math.sqrt etc..
import numpy as np #import numpy qu'on pourra appeler ainsi np.array([2,3,4]) par ex
import matplotlib.pyplot as plt

def my_round(x) :
  n = floor(x)
  decimal = floor(x * 10) % 10
  if decimal < 5 :
    return n
  return n+1

def rp(x,p):
    if x==0 or x==1:
        return x
    if x<1:
        n = 0
        while (floor(x) == 0):
            x *= 10
            n += 1
        x *= 10**(p-1)
        x  = my_round(x)
        x *= 10**(1-n-p)
    if x>1:
        n = 0
        while (floor(x) != 0):
            x /= 10
            n += 1
        x *= 10**p
        x  = my_round(x)
        x *= 10**(n-p)

    return x


def rp_str(x,p):
  x_str = str(x)
  [x_int_part, x_dec_part] = x_str.split(".")
  [int_len, dec_len] = [len(x_int_part), len(x_dec_part)]


  # Case 1 : precision only on the integer part
  if p <= int_len :
    res = int(x_int_part[0:p])
    
    if  p < int_len and int(x_int_part[p]) >= 5 :
      res += 1
    
    res = res * 10**(int_len - p)
    return res


  # Case 2 : int part is 0
  if int(x_int_part) == 0 :
    pos = 0
    while int(x_dec_part[pos]) == 0 :
      pos += 1

    res = float("0." + x_dec_part[0:pos+p])

    if  pos+p < dec_len and int(x_dec_part[pos+p]) >= 5 :
      res += 10**(-pos-p)
    
    return res


  # Case 3 : generic case
  res = float(x_int_part + "." + x_dec_part[0:p-int_len])

  if  p-int_len < dec_len and int(x_dec_part[p-int_len]) >= 5 :
    res += 10**(int_len-p)
  
  return res


def add_rp(x, y, p) :
  return rp(x,p) + rp(y,p)
  
def mul_rp(x, y, p) :
  return rp(x,p) * rp(y,p)


def relative_err_add(x, y, p) :
  num = abs( (x + y) - add_rp(x, y, p) )

  return num / abs(x + y)


def relative_err_mul(x, y, p) :
  num = abs( (x * y) - mul_rp(x, y, p) )

  return num / abs(x * y)


def my_log(p):
  i = 1
  res = 0
  err = inf
  log_2 = log(2)
  while err > 10**(-p) :
    res += ( (-1)**(i+1) ) / i
    i += 1
    err = abs(log_2 - rp(res, p)) / log_2
  
  return rp(res, p);

def plot_add(x, p):
  y = np.linspace(0,100,300)
  rel_err_add= [relative_err_add(x,i,p) for i in y]

  fig=plt.figure()
  plt.plot(y,rel_err_add)
  plt.xlabel("y")
  plt.ylabel("Erreur relative de la somme")
  plt.title("relative_err_add(x,y,p)")
  plt.savefig("img/relative_err_add.png",dpi=fig.dpi)

  print("Erreur relative maximale de la somme pour \nx = ",x ,
        " et p = ", p, " :\n", np.amax(rel_err_add))
  print()

  
def plot_mul(x, p):
  y = np.linspace(1,100,300)
  rel_err_mul= [relative_err_mul(x,i,p) for i in y]

  fig=plt.figure()
  plt.plot(y,rel_err_mul)
  plt.xlabel("y")
  plt.ylabel("Erreur relative de la multiplication")
  plt.title("relative_err_mul(x,y,p)")
  plt.savefig("img/relative_err_mul.png",dpi=fig.dpi)

  print("Erreur relative maximale de la multiplication pour \nx = ",x ,
        " et p = ", p, " :\n", np.amax(rel_err_mul))
  print()

  
def all_plot(x, p):
  y = np.linspace(1,100,300)
  rel_err_add= [relative_err_add(x,i,p) for i in y]
  rel_err_mul= [relative_err_mul(x,i,p) for i in y]

  fig=plt.figure()
  plt.plot(y,rel_err_add, label="addition")
  plt.plot(y,rel_err_mul, label="multiplication")
  plt.xlabel("y")
  plt.ylabel("Erreurs relatives")
  plt.title("Erreurs relatives")
  plt.legend()
  plt.savefig("img/relative_err_all.png",dpi=fig.dpi)

  print("pour x = ",x , " et p = ", p, " :\n")
  print("Erreur relative maximale de la somme :\n", np.amax(rel_err_add))
  print("Erreur relative maximale de la multiplication :\n", np.amax(rel_err_mul))
  print()

