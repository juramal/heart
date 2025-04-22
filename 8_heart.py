#%% BIBLIOTECAS
# instalar bibloteca Pandas
# python -m pip install pandas

import pandas as pd
from sklearn.neural_network import MLPClassifier

#%% CARGA DOS DADOS
df_heart = pd.read_csv('heart.csv')
print('Tabela de dados:\n', df_heart)
input('Aperte uma tecla para continuar: \n')

# matriz de treinamento (registros com campos ou atributos)
X = df_heart.loc[:, 'Age':'ST_Slope']   # de Age até ST_Slope
print("Matriz de entradas (treinamento):\n", X)
input('Aperte uma tecla para continuar: \n')

# vetor de classes
y = df_heart['HeartDisease']
print("Vetor de classes (treinamento):\n", y)
input('Aperte uma tecla para continuar: \n')

#%% ONE HOT ENCODER - pois os dados são nominais
from sklearn.preprocessing import OneHotEncoder

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X = encoder.fit_transform( df_heart.loc[:, 'Age':'ST_Slope'] )
print("Matriz de entradas codificadas:\n", X)
input('Aperte uma tecla para continuar: \n')

#%% CONFIG REDE NEURAL
mlp = MLPClassifier(verbose=True, 
                    max_iter=10000, 
                    tol=1e-6, 
                    activation='relu')

#%% TREINAMENTO DA REDE
mlp.fit(X,y)      # executa treinamento - ver console

#%% testes
print('\n')
for caso in X :
    print('caso: ', caso, ' previsto: ', mlp.predict([caso]) )

#%% teste de dado "não visto:"
z = pd.read_csv('heart_test.csv')

# Garantir que as colunas de teste sejam as mesmas usadas no treinamento
X_test = z.loc[:, 'Age':'ST_Slope']

# Transformar os dados de teste usando o encoder ajustado
X1 = encoder.transform(X_test)
print("\nNovo caso codificado (One-Hot Encoding):")
print(pd.DataFrame(X1))  # Exibir como DataFrame para melhor visualização
input('Aperte uma tecla para continuar: \n')

# previsão
predictions = mlp.predict(X1)
print("\nPrevisões para os novos casos:")
for i, pred in enumerate(predictions):
    print(f"Registro {i + 1}: {z.iloc[i].to_dict()} => Previsão: {pred}")

#%% ALGUNS PARÂMETROS DA REDE
print("\nResumo dos parâmetros da rede neural:")
print(f"Classes: {mlp.classes_}")  # lista de classes
print(f"Erro (Loss): {mlp.loss_:.6f}")  # fator de perda (erro)
print(f"Amostras visitadas: {mlp.t_}")  # número de amostras de treinamento visitadas
print(f"Atributos de entrada: {mlp.n_features_in_}")  # número de atributos de entrada
print(f"Número de ciclos (iterações): {mlp.n_iter_}")  # número de iterações no treinamento
print(f"Número de camadas: {mlp.n_layers_}")  # número de camadas da rede
print(f"Tamanhos das camadas ocultas: {mlp.hidden_layer_sizes}")
print(f"Número de neurônios de saída: {mlp.n_outputs_}")  # número de neurônios de saída
print(f"Função de ativação: {mlp.out_activation_}")  # função de ativação utilizada
