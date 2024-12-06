import argparse
import pandas as pd
from aproximacion import aproximacion

def main():
    # Configurar argparse para manejar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Ajustar y graficar un polinomio de grado especificado.')
    parser.add_argument('grado', type=int, help='Grado del polinomio a ajustar')
    
    # Parsear los argumentos
    args = parser.parse_args()
    
    # Leer los datos del archivo CSV
    data = pd.read_csv('mnist_accuracy.csv', encoding='latin1')
    
    # Llamar a la función de ajuste y graficado con el grado especificado
    aproximacion(data, args.grado)

if __name__ == "__main__":
    main()