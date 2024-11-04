import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('2-classificacao/EMGsDataset.csv', delimiter=',')

X = np.concatenate((
    data[:2, data[2, :]==1],
    data[:2, data[2, :]==2],
    data[:2, data[2, :]==3],
    data[:2, data[2, :]==4],
    data[:2, data[2, :]==5]
), axis=1)

P, N = X.shape

Y = np.concatenate((
    np.tile(0, (10000, 1)), 
    np.tile(1, (10000, 1)), 
    np.tile(2, (10000, 1)), 
    np.tile(3, (10000, 1)), 
    np.tile(4, (10000, 1))
)).reshape(1,N)

R = 500

percent = 0.8

treno = int(percent*N)

index = np.arange(N)

results = []

def gi(x_novo, media_i, inv):
    return (x_novo - media_i).T@inv@(x_novo-media_i)

for _ in range(R):
    np.random.shuffle(index)

    treno_i = index[:treno]
    tete_i = index[treno:]

    x_treno = X[:,treno_i]
    x_teste = X[:,tete_i]

    y_treno = Y[:,treno_i]
    y_teste = Y[:,tete_i]
    
    grupos_x = []
    matriz_vari = np.cov(x_treno)
    matriz_inv = np.linalg.inv(matriz_vari)
    
    for i in range(5):
        xni = x_treno[:, y_treno[0,:]==i]
        
        grupos_x.append(xni)
        
    medias = []
    
    for i in range(5):
        medias.append(np.mean(grupos_x[i], axis=1).reshape((2,1)))
        
    y_pred = []
    for i in range(x_teste.shape[1]):
        x_novo = x_teste[:,i].reshape((2,1))
        predict = []

        for j in range(5):
            predict.append(gi(x_novo, medias[j], matriz_inv))
        
        y_pred.append(np.argmin(predict))
    
    y_pred=np.array(y_pred)
    y_teste = y_teste[0]
    
    accuracy = np.mean(y_pred == y_teste)
    
    results.append(accuracy)

print(f"{'Média':<15} {'Desvio-Padrão':<20} {'Maior Valor':<15} {'Menor Valor':<15}")
print(f"{np.mean(results):<15.6e} {np.std(results):<20.6f} {np.max(results):<15.6e} {np.min(results):<15.6e}")