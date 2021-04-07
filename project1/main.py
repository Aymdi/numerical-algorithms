from algo_cordic import *
from repr_nombre_en_machine import *


print("----- PARTIE 1 -----\n")

n1 = 10507.1823
n2 = 0.0001857563
n3 = 2.252544

print("------ EX 1 ------\n")

print("rp(10507.1823, 3)",rp(n1, 3),sep=' = ')
print()
print("rp(10507.1823, 4)",rp(n1, 4),sep=' = ')
print()
print("rp(10507.1823, 5)",rp(n1, 5),sep=' = ')
print()

print("rp(0.0001857563, 4)",rp(n2, 4),sep=' = ')
print()
print("rp(0.0001857563, 6)",rp(n2, 6),sep=' = ')
print()

print("rp(np.pi, 4)",rp(np.pi, 4),sep=' = ')
print() 
print("rp(np.pi, 6)",rp(np.pi,6),sep=' = ')
print()


print("------ EX 2 / 3 ------\n")
# expected : 10507.2 + 2.25254 = 10509.45254
# without rounding :           = 10509.434844
print("add_rp(10507.1823, 2.252544, 6)",add_rp(n1, n3, 6),sep=' = ')
print("error : " , relative_err_add(n1, n3, 6))
print()

# expected : 10507.2 * 2.25254 = 23667.888288
# without rounding :           = 23667.4281307
print("mul_rp(10507.1823, 2.252544, 6)",mul_rp(n1, n3, 6),sep=' = ')
print("error : " , relative_err_mul(n1, n3, 6))
print()


print("------ EX 4 ------\n")

n = np.pi * (10**(-1))
p = 1

all_plot(n, p)


print("------ EX 5 ------\n")

print("my_log(6)",my_log(6),sep=' = ')
print("log(2)",rp(log(2), 6),sep=' = ')



###### PARTIE CORDIC ######


#F1
x = np.linspace(2, 100, 500)
y = [ ((log_cordic(i)-np.log(i))/np.abs(np.log(i))) for i in x ]
fig=plt.figure()
plt.plot(x,y)
plt.xlabel("Abscisses")
plt.ylabel("Erreur relative")
plt.title("Le logarithme népérien")
plt.savefig("img/er_ln.png",dpi=fig.dpi)

#F2
x = np.linspace(0, 100, 500)
y = [ ((exp_cordic(i)-np.exp(i))/np.abs(np.exp(i))) for i in x ]
fig=plt.figure()
plt.plot(x,y)
plt.xlabel("Abscisses")
plt.ylabel("Erreur relative")
plt.title("La fonction exponentielle")
plt.savefig("img/er_exp.png",dpi=fig.dpi)

#F3 #
x = np.linspace(0.01, 100, 500)
y = [ ((atan_cordic(i)-np.arctan(i))/(np.abs(np.arctan(i)))) for i in x ]
fig=plt.figure()
plt.plot(x,y)
plt.xlabel("Abscisses")
plt.ylabel("Erreur relative")
plt.title("L'arctangente")
plt.savefig("img/er_atan.png",dpi=fig.dpi)


#F4
x = np.linspace(0.01, 100, 500)
y = [ ((tan_cordic(i)-np.tan(i))/(np.abs(np.tan(i)))) for i in x ]
fig=plt.figure()
plt.plot(x,y)
plt.xlabel("Abscisses")
plt.ylabel("Erreur relative")
plt.title("La tangente")
plt.savefig("img/er_tan.png",dpi=fig.dpi)



print()
print("Tous les graphes d'erreurs relatives de la partie 1 et 2 se trouvent dans le dossier img.")
