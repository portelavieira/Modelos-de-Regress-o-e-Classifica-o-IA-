import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('2-classificacao/EMGsDataset.csv', delimiter=',')
data = data.T

X = np.concatenate((
    data[data[:,2]==1, :2],
    data[data[:,2]==2, :2],
    data[data[:,2]==3, :2],
    data[data[:,2]==4, :2],
    data[data[:,2]==5, :2]
))

N, P = X.shape

X = np.concatenate((
    np.ones((N, 1)), X
), axis=1)

Y = np.concatenate((
    np.tile(np.array([[1, -1, -1, -1, -1]]), (10000, 1)), 
    np.tile(np.array([[-1, 1, -1, -1, -1]]), (10000, 1)), 
    np.tile(np.array([[-1, -1, 1, -1, -1]]), (10000, 1)), 
    np.tile(np.array([[-1, -1, -1, 1, -1]]), (10000, 1)), 
    np.tile(np.array([[-1, -1, -1, -1, 1]]), (10000, 1))
))

R = 500

percent = 0.8

treno = int(percent*N)

index = np.arange(N)

results = []

for _ in range(R):
    np.random.shuffle(index)

    treno_i = index[:treno]
    tete_i = index[treno:]

    x_treno = X[treno_i]
    x_teste = X[tete_i]

    y_treno = Y[treno_i]
    y_teste = Y[tete_i]

    W = np.linalg.inv(x_treno.T @ x_treno) @ x_treno.T @ y_treno #treino

    Y_prev = x_teste @ W

    Y_prev = np.argmax(Y_prev, axis=1)
    y_teste = np.argmax(y_teste, axis=1)

    accuracy = np.mean(Y_prev == y_teste)
    
    results.append(accuracy)

print(f"{'Média':<15} {'Desvio-Padrão':<20} {'Maior Valor':<15} {'Menor Valor':<15}")
print(np.mean(results), np.std(results), np.max(results), np.min(results))