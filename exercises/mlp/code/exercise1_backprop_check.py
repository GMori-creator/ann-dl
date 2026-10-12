"""
Exercise 1 - conferência numérica da derivação feita à mão no relatório.

A derivação do relatório é feita em Markdown, passo a passo. Este script NÃO usa
nenhuma biblioteca de ML: só aritmética com math.tanh, escrita na mesma ordem das
contas do relatório, para conferir cada número (4+ casas decimais).
"""
import math

# --- valores do enunciado -------------------------------------------------
x = [0.5, -0.2]
y = 1.0
W1 = [[0.3, -0.1], [0.2, 0.4]]   # W1[j][i]: neurônio oculto j, entrada i
b1 = [0.1, -0.2]
W2 = [0.5, -0.3]                 # V
b2 = 0.2                         # b^y
eta = 0.3


def fmt(v):
    return [round(t, 6) for t in v] if isinstance(v, list) else round(v, 6)


# --- A: forward -----------------------------------------------------------
z1 = [W1[j][0] * x[0] + W1[j][1] * x[1] + b1[j] for j in range(2)]
h1 = [math.tanh(z) for z in z1]
u2 = W2[0] * h1[0] + W2[1] * h1[1] + b2
yhat = math.tanh(u2)
L = (y - yhat) ** 2
print("A) forward")
print("z1   =", fmt(z1))
print("h1   =", fmt(h1))
print("u2   =", fmt(u2))
print("yhat =", fmt(yhat))
print("L    =", fmt(L))

# --- B: backward ----------------------------------------------------------
dL_dyhat = -2 * (y - yhat)
dyhat_du2 = 1 - yhat ** 2
dL_du2 = dL_dyhat * dyhat_du2
dL_dW2 = [dL_du2 * h1[0], dL_du2 * h1[1]]
dL_db2 = dL_du2
dL_dh1 = [dL_du2 * W2[0], dL_du2 * W2[1]]
dh1_dz1 = [1 - h1[0] ** 2, 1 - h1[1] ** 2]
dL_dz1 = [dL_dh1[j] * dh1_dz1[j] for j in range(2)]
dL_dW1 = [[dL_dz1[j] * x[i] for i in range(2)] for j in range(2)]
dL_db1 = dL_dz1[:]
print("\nB) backward")
print("dL/dyhat   =", fmt(dL_dyhat))
print("dyhat/du2  =", fmt(dyhat_du2))
print("dL/du2     =", fmt(dL_du2))
print("dL/dW2     =", fmt(dL_dW2))
print("dL/db2     =", fmt(dL_db2))
print("dL/dh1     =", fmt(dL_dh1))
print("dh1/dz1    =", fmt(dh1_dz1))
print("dL/dz1     =", fmt(dL_dz1))
print("dL/dW1     =", [fmt(r) for r in dL_dW1])
print("dL/db1     =", fmt(dL_db1))

# --- C: update ------------------------------------------------------------
W2n = [W2[k] - eta * dL_dW2[k] for k in range(2)]
b2n = b2 - eta * dL_db2
W1n = [[W1[j][i] - eta * dL_dW1[j][i] for i in range(2)] for j in range(2)]
b1n = [b1[j] - eta * dL_db1[j] for j in range(2)]
print("\nC) update (eta = 0.3)")
print("W1 novo =", [fmt(r) for r in W1n])
print("b1 novo =", fmt(b1n))
print("W2 novo =", fmt(W2n))
print("b2 novo =", fmt(b2n))

# segunda passada forward com os parâmetros novos
z1n = [W1n[j][0] * x[0] + W1n[j][1] * x[1] + b1n[j] for j in range(2)]
h1n = [math.tanh(z) for z in z1n]
u2n = W2n[0] * h1n[0] + W2n[1] * h1n[1] + b2n
yhat_n = math.tanh(u2n)
L_n = (y - yhat_n) ** 2
print("\nsegunda passada: yhat =", fmt(yhat_n), " L =", fmt(L_n), " (antes:", fmt(L), ")")
