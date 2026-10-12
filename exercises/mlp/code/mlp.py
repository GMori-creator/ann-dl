"""
mlp.py - MLP implementado do zero (numpy), usado SEM alterações nos Exercícios 2, 3 e 4.

- Tamanhos de camada são um argumento: layer_sizes = [n_entrada, h1, ..., n_saida].
- Camadas ocultas: tanh.
- Saída: "sigmoid" (binário, 1 unidade) + entropia cruzada binária, ou
         "softmax" (multiclasse, K unidades) + entropia cruzada categórica.
- Em AMBOS os casos o gradiente da loss em relação à pré-ativação da saída é
      dL/du = (saída - Y) / N
  (derivado no relatório), então forward/backward/update são o mesmo código;
  só mudam o tamanho da saída, a função de saída/loss e os hiperparâmetros.
- Sem autodiff: todos os gradientes são derivados à mão e escritos em backward().
"""
import numpy as np


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def softmax(u):
    u = u - u.max(axis=1, keepdims=True)  # estabilidade numérica
    e = np.exp(u)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, layer_sizes, output, rng):
        assert output in ("sigmoid", "softmax")
        self.layer_sizes = list(layer_sizes)
        self.output = output
        # Inicialização Xavier para tanh: W ~ N(0, 1/fan_in), b = 0
        self.W = [rng.normal(0.0, np.sqrt(1.0 / m), size=(m, n))
                  for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(n) for n in layer_sizes[1:]]

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    # ---------------------------------------------------------------- forward
    def forward(self, X):
        """Retorna a saída (probabilidades) e as ativações de cada camada."""
        activations = [X]
        a = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            a = np.tanh(a @ W + b)
            activations.append(a)
        u = a @ self.W[-1] + self.b[-1]
        out = sigmoid(u) if self.output == "sigmoid" else softmax(u)
        return out, activations

    # ------------------------------------------------------------------- loss
    def loss(self, out, Y):
        eps = 1e-12
        if self.output == "sigmoid":   # BCE média
            return float(-np.mean(Y * np.log(out + eps) + (1 - Y) * np.log(1 - out + eps)))
        return float(-np.mean(np.sum(Y * np.log(out + eps), axis=1)))  # CCE média

    # --------------------------------------------------------------- backward
    def backward(self, out, activations, Y):
        N = Y.shape[0]
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        delta = (out - Y) / N                      # dL/du da camada de saída
        for l in range(len(self.W) - 1, -1, -1):
            grads_W[l] = activations[l].T @ delta  # dL/dW = a_{l-1}^T delta
            grads_b[l] = delta.sum(axis=0)         # dL/db = soma de delta
            if l > 0:
                dh = delta @ self.W[l].T           # dL/dh_{l-1}
                delta = dh * (1.0 - activations[l] ** 2)  # tanh' = 1 - tanh^2
        return grads_W, grads_b

    # ----------------------------------------------------------------- update
    def update(self, grads_W, grads_b, lr):
        for l in range(len(self.W)):
            self.W[l] -= lr * grads_W[l]
            self.b[l] -= lr * grads_b[l]

    # -------------------------------------------------------------------- fit
    def _targets(self, y):
        if self.output == "sigmoid":
            return y.reshape(-1, 1).astype(float)
        return np.eye(self.layer_sizes[-1])[y]     # one-hot

    def fit(self, X, y, lr, epochs):
        """Gradiente descendente em lote completo. Retorna a loss de treino por época."""
        Y = self._targets(y)
        history = []
        for _ in range(epochs):
            out, acts = self.forward(X)
            history.append(self.loss(out, Y))
            gW, gb = self.backward(out, acts, Y)
            self.update(gW, gb, lr)
        return history

    # ---------------------------------------------------------------- predict
    def predict(self, X):
        out, _ = self.forward(X)
        if self.output == "sigmoid":
            return (out[:, 0] >= 0.5).astype(int)
        return out.argmax(axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))
