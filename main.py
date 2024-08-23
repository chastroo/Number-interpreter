import cv2
import numpy as np
from sklearn import datasets
import pandas as pd

# Importamos el dataset
digitos = datasets.load_digits()

# Calculamos los promedios
df_promedios_completos = pd.DataFrame()  # DataFrame vacio donde colocaremos las matrices promediadas
separador = pd.DataFrame(data=[[0, 0, 0, 0, 0, 0, 0, 0]])

promedios = []

for digito in range(10):    # buscamos todas las matrices para cada uno de los targets y las guardamos en una lista
    matrices = []
    for i, j in enumerate(digitos['target']):
        if j == digito:
            matrices.append(digitos['data'][i])

    # Creamos un DataFrame con las matrices, que tiene una fila por cada matriz y una columna por cada pixel
    df_matrices = pd.DataFrame(matrices)
    # Usamos mean para calcular el promedio, axis = 0 para que tome los valores por columnas
    promedio = np.mean(df_matrices, axis=0)
    # Lo convertimos a un array para cambiar su forma a un 8x8
    promedio = np.array(promedio)
    promedio = promedio.reshape(8, 8)
    # Redondeamos los promeidos
    promedio = np.round(promedio)
    # Agregamos la matriz a la lista promedios
    promedios.append(promedio)
    # Agregamos la matriz obtenida a nuestro DataFrame, y colocamos debajo un separador
    df_promedio = pd.DataFrame(promedio)
    df_promedios_completos = pd.concat([df_promedios_completos, df_promedio], ignore_index=True)
    df_promedios_completos = pd.concat([df_promedios_completos, separador], ignore_index=True)


# Creamos un archivo .csv con los promedios de cada numero, para que se vea con claridad el resultado
df_promedios_completos.to_csv('promedio.csv')

# Cargamos la imagen y la reescalamos
imagen = cv2.imread('img8.jpg', cv2.IMREAD_GRAYSCALE)
imagen_pequenia = cv2.resize(src=imagen, dsize=(8, 8))


# Modificamos los valores de los pixeles de 0 a 16

for i in range(len(imagen_pequenia)):
    for j in range(len(imagen_pequenia)):
        imagen_pequenia[i][j] = 255 - imagen_pequenia[i][j]

for i in range(len(imagen_pequenia)):
    for j in range(len(imagen_pequenia)):
        imagen_pequenia[i][j] = imagen_pequenia[i][j] / 255 * 16

print(imagen_pequenia)


# Creamos un archivo .csv con el DataDrame del numero
numero = pd.DataFrame(imagen_pequenia)
numero.to_csv('numero_escrito.csv')


# Funcion que calcula la distancia euclidiana entre 2 matrices
def distancia_euclidiana(imagen, imagen2):
    suma = 0
    for lista in range(len(imagen)):
        for pixel in range(len(imagen[0])):
            suma += (imagen[lista][pixel] - imagen2[lista][pixel]) ** 2

    return round(suma ** 0.5, 2)

# Funcion que recibe las distancias obtenidas y encuentra las mas cercanas.
def calcular_vecinos_cercanos(distancias, num_vecinos):
    print("hola")
    distancias_cercanas = [_ for _ in sorted(distancias)[:num_vecinos]]
    indices = []
    for distancia in distancias_cercanas:
        indice = distancias.index(distancia)
        indices.append(indice)

    print(f'\nLas distancias calculadas para los {num_vecinos} vecinos mas cercanos son: {distancias_cercanas}, y sus indices son {indices}')

    targets = []
    for i, j in enumerate(digitos['target']):
        if i in indices:
            targets.append(j)

    print('\nLOS TARGETS SON', targets)
    if num_vecinos > 1:
        for t in targets:
            if targets.count(t) == 3 or targets.count(t) == 2:
                target_final = t
                return target_final
        num_vecinos += 1
        calcular_vecinos_cercanos(distancias, num_vecinos=num_vecinos)
    else:
        return targets[0]


# Inteligencia Artificial 1 - Utilizamos 3 vecinos

distancias = []
for i in range(1797):
    distancias.append(distancia_euclidiana(imagen_pequenia, digitos['images'][i]))

target_final = calcular_vecinos_cercanos(distancias, 3)

print(f'\nSOY LA INTELIGENCIA ARTIFICIAL 1, Y HE DETECTADO QUE EL DIGITO CORRESPONDE AL NUMERO: {target_final}')

# Inteligencia Artificial 2 - Utilizamos los promedios obtenidos y 1 vecino

distancias = []
for i in range(10):
    distancias.append(distancia_euclidiana(imagen_pequenia, promedios[i]))

target = calcular_vecinos_cercanos(distancias, 1)
print(f'\nSOY LA INTELIGENCIA ARTIFICIAL 2, Y HE DETECTADO QUE EL DIGITO CORRESPONDE AL NUMERO: {target}')
