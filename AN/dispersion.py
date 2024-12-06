import matplotlib.pyplot as plt
import pandas as pd

# Leer los datos del archivo CSV
data = pd.read_csv('mnist_accuracy.csv')

# Crear el gráfico de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(data['Tamaño Dataset'], data['Precision'], color='blue', label='Precision')

# Añadir títulos y etiquetas
plt.title('Precision vs Tamaño Dataset')
plt.xlabel('Tamaño Dataset')
plt.ylabel('Precision')
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()