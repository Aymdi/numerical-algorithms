import matplotlib.pyplot as plt
import numpy as np
#Les problème de Cauchy sont sous la forme ([y0, t0], f), les f sont de la forme y' = f(y, t)

def step_Euler(y, t, h, f):
  return y + h * f(y, t)

def step_point_milieu(y, t, h, f):
  return y + h*f(y, t + h / 2)

def step_Runge_Kutta(y, t, h, f):
  return y + 1/6*h*(f(y,t)+2*f(y+1/2*h*f(y,t),t+h/2)+2*f(y+1/2*h*f(y+1/2*h*f(y,t),t+h/2),t+h/2)+f(y+1/2*h*f(y+1/2*h*f(y+1/2*h*f(y,t),t+h/2),t+h/2),t+1/2*h))

def step_Heun(y, t, h, f):
  return y + h*(f(y,t) + f(step_Euler(y,t,h,f),t+h))/2


def meth_n_step(y0, t0, N, h, f, meth):
  y, t = [y0], [t0]
  dim = len(y0)
  for i in range(N):
    y.append(meth(y[-1], t[-1], h, f))
    t.append(t[-1] + h)
  return y, t

def norme(yn_i, y2n_i):
  somme = 0
  for k in range(len(yn_i)):
    somme += (yn_i[k] - y2n_i[k])**2
  return math.sqrt(somme)

def norme_diff(yn, y2n, n):
  max = 0
  for i in range(n+1):
    temp = norme(yn[i], y2n[2*i])
    if temp > max:
      max = temp
  return max

def meth_epsilon(y0, t0, tf, eps, f, meth):
  N = 100
  h = (tf - t0) / N
  yn, tn = meth_n_step(y0, t0, N, h, f, meth)
  y2n, t2n = meth_n_step(y0, t0, 2*N, h/2, f, meth)
  #diff = norme_diff(yn, y2n, N)
  y2n = np.array(y2n)
  yn = np.array(yn)
  X = y2n[::2] - yn
  diff = np.linalg.norm(X)
  while diff > eps:
    N *= 2
    print(N)
    h = (tf - t0) / N
    yn, tn = y2n, t2n
    y2n, t2n = meth_n_step(y0, t0, N, h, f, meth)
    y2n = np.array(y2n)
    yn = np.array(yn)
    X = y2n - yn
    diff = np.linalg.norm(X)
  return np.array(yn), np.array(tn)

def meth_tangente(y0, t0, tf, eps, f, meth):
  table_y = []
  table_t = []
  for i in range(len(y0)):
    y, t = meth_epsilon(y0[i], t0[i], tf, eps, f, meth)
    table_y.append(y)
    table_t.append(t)
  for i in range(len(y0)):
    for j in range(len(table_y[i])):
      for k in range(len(table_y[i][j])):
        plt.quiver(table_t[i][j], table_y[i][j][k], 1, f(table_y[i][j],table_t[i][j])[k])

""" Test """

#dimension 1:
def f_dim1(y, t):
  return y / (1 + t**2)
def y_exact_dim1(x):
  return np.array([math.exp(math.atan(x))])


PbCauchy = ([np.array([1]), 0], f_dim1)
'''
y_euler, t_euler = meth_epsilon(PbCauchy[0][0], PbCauchy[0][1], 5, 0.001, PbCauchy[1], step_Euler)
y_point_milieu, t_point_milieu = meth_epsilon(PbCauchy[0][0], PbCauchy[0][1], 5, 0.001, PbCauchy[1], step_point_milieu)
y_Runge_Kutta, t_Runge_Kutta = meth_epsilon(PbCauchy[0][0], PbCauchy[0][1], 5, 0.001, PbCauchy[1], step_Runge_Kutta)
y_Heun, t_Heun = meth_epsilon(PbCauchy[0][0], PbCauchy[0][1], 5, 0.001, PbCauchy[1], step_Heun)

y_dim1 = [y_exact_dim1(i) for i in t_euler]

plt.plot(t_euler, y_dim1, label="Solution exacte", color="k")
plt.plot(t_euler, y_euler, label="Méthode d'Euler", color="b")
plt.plot(t_point_milieu,y_point_milieu, label="Méthode du point-milieu", color = "g")
plt.plot(t_Runge_Kutta,y_Runge_Kutta, label="Méthode de Runge-Kutta", color="r")
plt.plot(t_Heun,y_Heun, label="Méthode de Heun", color='y')

#meth_tangente([-2,-1,0,1,2], [0,0,0,0,0], 5, 0.1, PbCauchy[1], step_Euler)
plt.show()
'''

## Dimension 2:

def f_dim2(y, t):
  return np.array([-y[1], y[0]])
def y_exact_dim2(t):
  return [math.cos(t), math.sin(t)]

PbCauchy2 = ([np.array([1, 0]), 0], f_dim2)
'''
y_euler, t_euler= meth_epsilon(PbCauchy2[0][0], PbCauchy2[0][1], 5, 0.01, PbCauchy2[1], step_Euler)
y_point_milieu, t_point_milieu = meth_epsilon(PbCauchy2[0][0], PbCauchy2[0][1], 5, 0.001, PbCauchy2[1], step_point_milieu)
y_Runge_Kutta, t_Runge_Kutta = meth_epsilon(PbCauchy2[0][0], PbCauchy2[0][1], 5, 0.001, PbCauchy2[1], step_Runge_Kutta)
y_Heun, t_Heun = meth_epsilon(PbCauchy2[0][0], PbCauchy2[0][1], 5, 0.001, PbCauchy2[1], step_Heun)

y_dim2 = [y_exact_dim2(i) for i in t_euler_dim2]

plt.plot(t_euler, y_dim2, label="Solution exacte", color="k")
plt.plot(t_euler, y_euler, label="Méthode d'Euler", color="b")
plt.plot(t_point_milieu,y_point_milieu, label="Méthode du point-milieu", color = "g")
plt.plot(t_Runge_Kutta,y_Runge_Kutta, label="Méthode de Runge-Kutta", color="r")
plt.plot(t_Heun,y_Heun, label="Méthode de Heun", color='y')

#meth_tangente([[2,0],[1,1],[1,0],[0,0],[-1,0],[-1,-1],[-2,0]],[0,0,0,0,0,0,0], 5, 0.2, PbCauchy2[1], step_Runge_Kutta)
plt.show()
'''
#Mesure des vitesses de cv

def tracer_vitesse_cv_ref(Pb_Cauchy, log2N, tf):
    diff_euler = []
    diff_pt_milieu = []
    diff_heun = []
    diff_rk4 = []
    tab_N = [2]
    y0 = Pb_Cauchy[0][0]
    t0 = Pb_Cauchy[0][1]
    f = Pb_Cauchy[1]
    h = (tf-t0) / 2
    yn_euler, tn_euler = meth_n_step(y0, t0, 2, h, f, step_Euler)
    y2n_euler, t2n_euler = meth_n_step(y0, t0, 4, h/2, f, step_Euler)

    yn_pt_milieu, tn_pt_milieu = meth_n_step(y0, t0, 2, h, f, step_point_milieu)
    y2n_pt_milieu, t2n_pt_milieu = meth_n_step(y0, t0, 4, h/2, f, step_point_milieu)

    yn_heun, tn_heun = meth_n_step(y0, t0, 2, h, f, step_Heun)
    y2n_heun, t2n_heun = meth_n_step(y0, t0, 4, h/2, f, step_Heun)

    yn_rk4, tn_rk4 = meth_n_step(y0, t0, 2, h, f, step_Runge_Kutta)
    y2n_rk4, t2n_rk4 = meth_n_step(y0, t0, 4, h/2, f, step_Runge_Kutta)

    diff_euler.append(norme_diff(yn_euler, y2n_euler, 2))
    diff_pt_milieu.append(norme_diff(yn_pt_milieu, y2n_pt_milieu, 2))
    diff_heun.append(norme_diff(yn_heun, y2n_heun, 2))
    diff_rk4.append(norme_diff(yn_rk4, y2n_rk4, 2))

    for i in range(3, log2N + 1):
        tab_N.append(2**i)
        h = (tf - t0) / (2**i)
        yn_euler, tn_euler = y2n_euler, t2n_euler
        y2n_euler, t2n_euler = meth_n_step(y0, t0, 2**i, h, f, step_Euler)

        yn_pt_milieu, tn_pt_milieu = y2n_pt_milieu, t2n_pt_milieu
        y2n_pt_milieu, t2n_pt_milieu = meth_n_step(y0, t0, 2**i, h, f, step_point_milieu)

        yn_heun, tn_heun = y2n_heun, t2n_heun
        y2n_heun, t2n_heun = meth_n_step(y0, t0, 2**i, h, f, step_Heun)

        yn_rk4, tn_rk4 = y2n_rk4, t2n_rk4
        y2n_rk4, t2n_rk4 = meth_n_step(y0, t0, 2**i, h, f, step_Runge_Kutta)

        diff_euler.append(norme_diff(yn_euler, y2n_euler, 2**(i-1)))
        diff_pt_milieu.append(norme_diff(yn_pt_milieu, y2n_pt_milieu, 2**(i-1)))
        diff_heun.append(norme_diff(yn_heun, y2n_heun, 2**(i-1)))
        diff_rk4.append(norme_diff(yn_rk4, y2n_rk4, 2**(i-1)))

    plt.plot(tab_N, diff_euler, label='convergence d\'Euler', color='b')
    plt.plot(tab_N, diff_pt_milieu, label='convergence du point milieu', color='g')
    plt.plot(tab_N, diff_rk4, label='convergence de Runge-Kutta', color='r')
    plt.plot(tab_N, diff_heun, label='convergence de Heun', color='m')
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.show()
if __name__ == "__main__":

    tracer_vitesse_cv_ref(PbCauchy, 14, 5)
