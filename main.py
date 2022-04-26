'''
    Transformando a temperatura de um grau Celsius para Fahrenheit utilizando
uma Rede Neural com tensorFlow
'''
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando o database necessário
temperatura_df = pd.read_csv('Database/Celsius-to-Fahrenheit.csv')
temperatura_df.reset_index(drop = True, inplace= True)

#sns.scatterplot(temperatura_df['Celsius'], temperatura_df['Fahrenheit'])
#plt.show()

# Definindo os dados de treinamento
x_train = temperatura_df['Celsius']
y_train = temperatura_df['Fahrenheit']

# Criando o modelo de rede neural
model = tf.keras.Sequential()
# O modelo tem uma entrada e uma saída
model.add(tf.keras.layers.Dense(units = 1, input_shape = [1]))

# Modificar o valor do 'Adam' para melhor resultado
model.compile(optimizer = tf.keras.optimizers.Adam(0.6), loss ='mean_squared_error')

# Fazendo o treinamento
epochs_hist = model.fit(x_train, y_train, epochs = 1000)

# Plotando um gráfico representando a evolução do erro
plt.plot(epochs_hist.history['loss'])
plt.show()

# Imprimindo o valor de peso encontrado
print(f"Peso: {model.get_weights()}")

# Testando, para verificar se ele faz a conversão de maneira correta
temp_c = 0
temp_f = model.predict([temp_c])
print(f"C = {temp_c}, F = {temp_f}")