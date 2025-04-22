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
encoder = OneHotEncoder(sparse_output=False)
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
X = encoder.fit_transform( z.loc[:, 'Age':'ST_Slope'] )

# X1 = encoder.fit_transform([X])  Não ajustar (fit) para o novo dado!!!
# Em vez, utilize o encoder já gerado:
X1 = encoder.transform(X)
print("\nNovo caso codificado: ", X1)
input('Aperte uma tecla para continuar: \n')

# previsão
print( X, '=', mlp.predict(X1) )
print("\n")

#%% ALGUNS PARÂMETROS DA REDE
print("Classes = ", mlp.classes_)     # lista de classes
print("Erro = ", mlp.loss_)        # fator de perda (erro)
print("Amostras visitadas = ", mlp.t_)           # número de amostras de treinamento visitadas 
print("Atributos de entrada = ", mlp.n_features_in_)   # número de atributos de entrada (campos de X)
print("N ciclos = ", mlp.n_iter_)      # númerode iterações no treinamento
print("N de camadas = ", mlp.n_layers_)    # número de camadas da rede
print("Tamanhos das camadas ocultas: ", mlp.hidden_layer_sizes)
print("N de neurons saida = ", mlp.n_outputs_)   # número de neurons de saida
print("F de ativação = ", mlp.out_activation_)  # função de ativação utilizada
