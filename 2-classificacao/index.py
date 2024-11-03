import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar os dados e organizar X e Y
data = np.loadtxt('2-classificacao/EMGsDataset.csv', delimiter=',', skiprows=1)
X = data[:, :2]  # Variáveis do sensor (Sensor 1 e Sensor 2)
Y = data[:, 2].astype(int)   # Classes

# Configurações para a simulação de Monte Carlo
R = 500
resultados = {
    'MQO tradicional': [],
    'Classificador Gaussiano Tradicional': [],
    'Classificador Gaussiano (Cov. de todo cj. treino)': [],
    'Classificador Gaussiano (Cov. Agregada)': [],
    'Classificador de Bayes Ingênuo': [],
    'Classificador Gaussiano Regularizado (λ=0.25)': [],
    'Classificador Gaussiano Regularizado (λ=0.5)': [],
    'Classificador Gaussiano Regularizado (λ=0.75)': [],
}

# Função para dividir dados em treino e teste
def split_train_test(X, Y, test_size=0.2):
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_set_size = int(n_samples * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]

# Função para calcular a acurácia
def calcular_acuracia(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Implementação dos modelos gaussianos
def classificador_gaussiano(X_train, y_train, X_test, cov_reg=0):
    classes = np.unique(y_train)
    mean_vectors = {c: X_train[y_train == c].mean(axis=0) for c in classes}
    cov_matrices = {c: np.cov(X_train[y_train == c].T) + cov_reg * np.eye(X_train.shape[1]) for c in classes}
    y_pred = []
    for x in X_test:
        probs = []
        for c in classes:
            mean = mean_vectors[c]
            cov = cov_matrices[c]
            prob = -0.5 * np.log(np.linalg.det(cov)) - 0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)
            probs.append(prob)
        y_pred.append(classes[np.argmax(probs)])
    return np.array(y_pred)

# Implementação do classificador de MQO tradicional
def mqo_classificador(X_train, y_train, X_test):
    unique_classes = np.unique(y_train)
    mean_vectors = {c: X_train[y_train == c].mean(axis=0) for c in unique_classes}
    y_pred = []
    for x in X_test:
        distances = [np.linalg.norm(x - mean_vectors[c]) for c in unique_classes]
        y_pred.append(unique_classes[np.argmin(distances)])
    return np.array(y_pred)

# 2. Visualização inicial dos dados (Gráfico de Dispersão)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', s=10)
plt.title('Gráfico de Dispersão dos Dados')
plt.xlabel('Sensor 1')
plt.ylabel('Sensor 2')
plt.colorbar(label='Classes')
plt.show()

# 5. Simulação de Monte Carlo com R = 500
lambdas = [0.25, 0.5, 0.75]
for _ in range(R):
    X_train, X_test, y_train, y_test = split_train_test(X, Y, test_size=0.2)

    # MQO Tradicional
    y_pred = mqo_classificador(X_train, y_train, X_test)
    resultados['MQO tradicional'].append(calcular_acuracia(y_test, y_pred))

    # Classificador Gaussiano Tradicional
    y_pred = classificador_gaussiano(X_train, y_train, X_test, cov_reg=0)
    resultados['Classificador Gaussiano Tradicional'].append(calcular_acuracia(y_test, y_pred))

    # Classificador Gaussiano (Cov. de todo cj. treino)
    y_pred = classificador_gaussiano(X_train, y_train, X_test, cov_reg=0)
    resultados['Classificador Gaussiano (Cov. de todo cj. treino)'].append(calcular_acuracia(y_test, y_pred))

    # Classificador Gaussiano (Cov. Agregada)
    cov_agregada = np.cov(X_train.T)
    y_pred = classificador_gaussiano(X_train, y_train, X_test, cov_reg=cov_agregada)
    resultados['Classificador Gaussiano (Cov. Agregada)'].append(calcular_acuracia(y_test, y_pred))

    # Classificador de Bayes Ingênuo (assumindo covariância diagonal)
    cov_diag = np.diag(np.var(X_train, axis=0))
    y_pred = classificador_gaussiano(X_train, y_train, X_test, cov_reg=cov_diag)
    resultados['Classificador de Bayes Ingênuo'].append(calcular_acuracia(y_test, y_pred))

    # Classificadores Gaussianos Regularizados (λ = 0.25, 0.5, 0.75)
    for lamb in lambdas:
        y_pred = classificador_gaussiano(X_train, y_train, X_test, cov_reg=lamb)
        resultados[f'Classificador Gaussiano Regularizado (λ={lamb})'].append(calcular_acuracia(y_test, y_pred))

# 6. Cálculo da média, desvio-padrão, maior e menor valor
resumo = {}
for modelo, acc in resultados.items():
    resumo[modelo] = {
        'Média': np.mean(acc),
        'Desvio-Padrão': np.std(acc),
        'Maior Valor': np.max(acc),
        'Menor Valor': np.min(acc)
    }

# Exibir resultados
print("Resultados de Acurácia:")
for modelo, metricas in resumo.items():
    print(f"{modelo}:")
    for metrica, valor in metricas.items():
        print(f"  {metrica}: {valor:.4f}")
    print()

# Gráficos dos resultados
labels = list(resumo.keys())
means = [metricas['Média'] for metricas in resumo.values()]
stds = [metricas['Desvio-Padrão'] for metricas in resumo.values()]

x = np.arange(len(labels))
plt.bar(x, means, yerr=stds, capsize=5)
plt.xticks(x, labels, rotation=45, ha='right')
plt.ylabel('Acurácia Média')
plt.title('Acurácia Média com Desvio-Padrão para Cada Modelo')
plt.tight_layout()
plt.show()