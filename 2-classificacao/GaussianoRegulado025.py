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

lamb = 0.000000000001

treno = int(percent*N)

index = np.arange(N)

results = []

def gi(x_novo, media_i, determ, inv):
    if lamb == 1:
        return -(x_novo - media_i).T@inv@(x_novo-media_i)
    else:
        return -(1/2)*np.log(determ)-(1/2)*(x_novo - media_i).T@inv@(x_novo-media_i)

for _ in range(R):
    np.random.shuffle(index)

    treno_i = index[:treno]
    tete_i = index[treno:]

    x_treno = X[:,treno_i]
    x_teste = X[:,tete_i]

    y_treno = Y[:,treno_i]
    y_teste = Y[:,tete_i]
    
    grupos_x = []
    matrizes_vari = []
    
    for i in range(5):
        xni = x_treno[:, y_treno[0,:]==i]
        
        grupos_x.append(xni)
        
        matrizes_vari.append(np.cov(xni))
        
    medias = []
    
    for i in range(5):
        medias.append(np.mean(grupos_x[i], axis=1).reshape((2,1)))
        
    m_agreg = np.zeros((P,P))
    
    for i in range(5):
        ni = grupos_x[i].shape[1]
        m_agreg += ni/(N*percent)*matrizes_vari[i]
        
    m_cov_fried = []
    for i in range(5):
        ni = grupos_x[i].shape[1]
        n = N*percent
        dividendo = (1 - lamb)*(ni*matrizes_vari[i])+(lamb*n*m_agreg)
        divisor = (1 - lamb)*ni+(lamb*n)
        
        m_cov_fried.append(dividendo/divisor)
        
    grupos_determ = []
    grupos_inver = []
    
    for i in range(5):
        grupos_determ.append(np.linalg.det(m_cov_fried[i]))
        grupos_inver.append(np.linalg.inv(m_cov_fried[i]))
        
    y_pred = []
    for i in range(x_teste.shape[1]):
        x_novo = x_teste[:,i].reshape((2,1))
        predict = []

        for j in range(5):
            predict.append(gi(x_novo, medias[j], grupos_determ[j], grupos_inver[j]))
        
        y_pred.append(np.argmax(predict))
    
    y_pred=np.array(y_pred)
    y_teste = y_teste[0]
    
    
    accuracy = np.mean(y_pred == y_teste)
    
    results.append(accuracy)

print(f"{'Média':<15} {'Desvio-Padrão':<20} {'Maior Valor':<15} {'Menor Valor':<15}")
print(f"{np.mean(results):<15.6e} {np.std(results):<20.6f} {np.max(results):<15.6e} {np.min(results):<15.6e}")