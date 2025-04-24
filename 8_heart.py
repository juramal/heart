#%% BIBLIOTECAS
# instalar bibloteca Pandas
# python -m pip install pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#%% CARGA DOS DADOS
df_heart = pd.read_csv('heart.csv')
print('Tabela de dados:\n', df_heart)
input('Aperte uma tecla para continuar: \n')

# Separar matriz de treinamento (X) e vetor de classes (y)
X = df_heart.loc[:, 'Age':'ST_Slope']   # de Age até ST_Slope
y = df_heart['HeartDisease']

# Dividir os dados em 80% para treinamento e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dados de treinamento e teste separados.")
input('Aperte uma tecla para continuar: \n')

#%% ONE HOT ENCODER - pois os dados são nominais
# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

print("Matriz de entradas codificadas (treinamento):\n", X_train_encoded)
input('Aperte uma tecla para continuar: \n')

#%% CONFIG REDE NEURAL
mlp = MLPClassifier(
                    verbose=True, 
                    max_iter=10000, 
                    tol=1e-6, 
                    activation='relu',
                    hidden_layer_sizes=(2, 6),
                    )

#%% TREINAMENTO DA REDE
mlp.fit(X_train_encoded, y_train)  # Executa treinamento

#%% TESTES
predictions = mlp.predict(X_test_encoded)

# Calcular e exibir a acurácia
accuracy = accuracy_score(y_test, predictions)
print(f"\nAcurácia do modelo: {accuracy * 100:.2f}%")

# Plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=mlp.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.show()

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