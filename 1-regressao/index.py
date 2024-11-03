import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados
data = np.loadtxt('1-regressao/aerogerador.dat')
X_raw = data[:, 0]  # Velocidade do vento
y = data[:, 1]      # Potência gerada

# Visualizar a relação entre velocidade do vento e potência gerada
plt.scatter(X_raw, y, alpha=0.5)
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.title('Gráfico de Dispersão - Velocidade do Vento vs Potência Gerada')
plt.show()

# Adiciona uma coluna de 1's para o intercepto
X = np.vstack([np.ones_like(X_raw), X_raw]).T

# Função para MQO Tradicional
def mqo_tradicional(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Função para MQO Regularizado
def mqo_regularizado(X, y, lambd):
    I = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + lambd * I) @ X.T @ y

# Média dos valores observáveis
media_observaveis = np.mean(y)

# Parâmetros de Monte Carlo
R = 500  # Número de simulações
lambdas = [0, 0.25, 0.5, 0.75, 1]  # Valores de lambda para o modelo regularizado

# Listas para armazenar os RSS
rss_tradicional = []
rss_regularizado = {lambd: [] for lambd in lambdas}
rss_media = []

# Função para calcular RSS
def calcular_rss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

np.random.seed(42)  # Para reprodutibilidade

# Loop de Monte Carlo
for _ in range(R):
    # Divisão treino-teste (80%-20%)
    indices = np.random.permutation(len(y))
    treino_size = int(0.8 * len(y))
    indices_treino, indices_teste = indices[:treino_size], indices[treino_size:]
    
    X_treino, X_teste = X[indices_treino], X[indices_teste]
    y_treino, y_teste = y[indices_treino], y[indices_teste]
    
    # Modelo MQO Tradicional
    beta_tradicional = mqo_tradicional(X_treino, y_treino)
    y_pred_tradicional = X_teste @ beta_tradicional
    rss_tradicional.append(calcular_rss(y_teste, y_pred_tradicional))
    
    # Modelo MQO Regularizado para cada lambda
    for lambd in lambdas:
        beta_regularizado = mqo_regularizado(X_treino, y_treino, lambd)
        y_pred_regularizado = X_teste @ beta_regularizado
        rss_regularizado[lambd].append(calcular_rss(y_teste, y_pred_regularizado))
    
    # Modelo Média dos Valores Observáveis
    y_pred_media = np.full_like(y_teste, media_observaveis)
    rss_media.append(calcular_rss(y_teste, y_pred_media))

# Função para calcular estatísticas
def calcular_estatisticas(rss_values):
    return {
        'Média': np.mean(rss_values),
        'Desvio-Padrão': np.std(rss_values),
        'Maior Valor': np.max(rss_values),
        'Menor Valor': np.min(rss_values)
    }

# Estatísticas para cada modelo
estatisticas = {
    'MQO Tradicional': calcular_estatisticas(rss_tradicional),
    'Média da Variável Dependente': calcular_estatisticas(rss_media)
}

# Estatísticas para os modelos regularizados
for lambd in lambdas:
    estatisticas[f'MQO Regularizado ({lambd})'] = calcular_estatisticas(rss_regularizado[lambd])

# Exibir as estatísticas em uma tabela
import pandas as pd

df_estatisticas = pd.DataFrame(estatisticas).T
print(df_estatisticas)

# Visualizar as estatísticas em um gráfico de barras
df_estatisticas[['Média', 'Desvio-Padrão', 'Maior Valor', 'Menor Valor']].plot(kind='bar', figsize=(10, 6))
plt.ylabel('RSS')
plt.title('Estatísticas do RSS para cada Modelo')
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.show()
