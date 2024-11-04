import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.loadtxt('1-regressao/aerogerador.dat')
X_raw = data[:, 0]
y = data[:, 1]      

plt.scatter(X_raw, y, alpha=0.5, color='hotpink')
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.title('Gráfico de Dispersão - Velocidade do Vento vs Potência Gerada')
plt.show()

X = np.concatenate([np.ones((X_raw.shape[0], 1)), X_raw.reshape(-1, 1)], axis=1)

def mqo_tradicional(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def mqo_regularizado(X, y, lambd):
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + lambd * I) @ X.T @ y

media_obs = np.mean(y)

R = 500  
lambdas = [0, 0.25, 0.5, 0.75, 1]

rss_tradicional = []
rss_regularizado = {lambd: [] for lambd in lambdas}
rss_media = []

index = np.arange(len(y))

def calcular_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

for _ in range(R):
    np.random.shuffle(index)
    treino_size = int(0.8 * len(y))
    index_treino, index_teste = index[:treino_size], index[treino_size:]
    
    X_treino, X_teste = X[index_treino], X[index_teste]
    y_treino, y_teste = y[index_treino], y[index_teste]
    
    beta_tradicional = mqo_tradicional(X_treino, y_treino)
    y_pred_tradicional = X_teste @ beta_tradicional
    rss_tradicional.append(calcular_rss(y_teste, y_pred_tradicional))
    
    for lambd in lambdas:
        beta_regularizado = mqo_regularizado(X_treino, y_treino, lambd)
        y_pred_regularizado = X_teste @ beta_regularizado
        rss_regularizado[lambd].append(calcular_rss(y_teste, y_pred_regularizado))
    
    y_pred_media = np.full_like(y_teste, media_obs)
    rss_media.append(calcular_rss(y_teste, y_pred_media))

def calcular_estatisticas(rss_tradicional, rss_regularizado, rss_media, lambdas):
    estatisticas = {
        'MQO Tradicional': {
            'Média': np.mean(rss_tradicional),
            'Desvio-Padrão': np.std(rss_tradicional),
            'Maior Valor': np.max(rss_tradicional),
            'Menor Valor': np.min(rss_tradicional)
        },
        'Média da Variável Dependente': {
            'Média': np.mean(rss_media),
            'Desvio-Padrão': np.std(rss_media),
            'Maior Valor': np.max(rss_media),
            'Menor Valor': np.min(rss_media)
        }
    }
    for lambd in lambdas:
        estatisticas[f'MQO Regularizado ({lambd})'] = {
            'Média': np.mean(rss_regularizado[lambd]),
            'Desvio-Padrão': np.std(rss_regularizado[lambd]),
            'Maior Valor': np.max(rss_regularizado[lambd]),
            'Menor Valor': np.min(rss_regularizado[lambd])
        }
    return estatisticas

estatisticas = calcular_estatisticas(rss_tradicional, rss_regularizado, rss_media, lambdas)
df_estatisticas = pd.DataFrame(estatisticas).T
print(df_estatisticas)

df_estatisticas[['Média', 'Desvio-Padrão', 'Maior Valor', 'Menor Valor']].plot(kind='bar', figsize=(10, 6))
plt.ylabel('RSS')
plt.title('Estatísticas do RSS para cada Modelo')
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.show()