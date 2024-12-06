import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import csv

# Descargar el dataset MNIST
print("Descargando el dataset MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Convertir etiquetas a enteros
y = y.astype(int)

# Dividir en entrenamiento 80% y prueba 20%
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# Tamaños de dataset para probar
dataset_sizes = [1000 * i for i in range(1, 21)]

# Crear archivo CSV
csv_file = "mnist_accuracy.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Tamaño Dataset", "Precision"])

# Entrenar y evaluar el modelo para diferentes tamaños de dataset
for size in dataset_sizes:
    X_train = X_train_full[:size]
    y_train = y_train_full[:size]
    
    # Entrenar un modelo de regresión logística
    model = LogisticRegression(max_iter=2000)  # Aumentar el número de iteraciones
    model.fit(X_train, y_train)
    
    # Predecir y calcular precisión
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Guardar resultados en el CSV
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([size, accuracy])
    
    print(f"Tamaño del dataset: {size}, Precisión: {accuracy:.4f}")

print(f"Resultados guardados en {csv_file}")