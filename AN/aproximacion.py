import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score



def aproximacion(data,grado):
    # Leemos datos!
   

    #Grafico
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Tamaño Dataset'], data['Precision'], color='blue', label='Precision')

    #Titulos
    plt.title('Precision vs Tamaño Dataset')
    plt.xlabel('Tamaño Dataset')
    plt.ylabel('Precision')

    # Ajustar una línea de mínimos cuadrados
    coefficients = np.polyfit(data['Tamaño Dataset'], data['Precision'], grado) #<--- np.polyfit(x, y, grado del polinomio)
    polynomial = np.poly1d(coefficients)
    x_fit = np.linspace(data['Tamaño Dataset'].min(), data['Tamaño Dataset'].max(), 100)
    y_fit = polynomial(x_fit)

    # Bondad del ajuste
    y_pred = polynomial(data['Tamaño Dataset'])
    r2 = r2_score(data['Precision'], y_pred)

    # Graficamos la funcion envolvente 
    plt.plot(x_fit, y_fit, color='red', label=f' Bondad (R^2 = {r2:.4f})')


    plt.legend()
    plt.grid(True)
    plt.show()

    ##Polinomio generado
    print("Polinomio generado:")
    print(polynomial)
