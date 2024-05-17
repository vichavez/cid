import numpy as np
class PolynomialRegression:
    #Se define una nueva clase llamada PolynomialRegression.
    def __init__(self, degree):
        #Se define un método especial _init_ que inicializa una instancia de la clase. Toma un parámetro 
        #degree, que representa el grado del polinomio.
        self.degree = degree
        #Se asigna el grado del polinomio a un atributo de la instancia llamado degree.
        #degree es un parámetro que permite controlar el grado del polinomio en el modelo de regresión 
        #polinomial. Un grado más alto generalmente permite que el modelo se ajuste mejor a los datos, 
        #pero también puede aumentar el riesgo de sobreajuste. Por lo tanto, es importante elegir un 
        #grado apropiado según la naturaleza de los datos y el problema específico.
    def ajuste(self, x, y):
        #Se define un método llamado ajuste, que ajusta el modelo de regresión polinomial a los datos de entrada X e y.
        n = len(x)
        #Se calcula el número de muestras en el conjunto de datos X.
        x_poly = self.caracteristicas_polinomiales(x)
        #Se llama a un método interno caracteristicas_polinomiales para obtener las características polinomiales de X.
        self.coeficientes = self.coeficientes_calculados(x_poly, y)
        #Se calculan los coeficientes del modelo de regresión polinomial llamando al método _calculate_coefficients.
        
    def prediccion(self, X):
        #Se define un método llamado prediccion, que realiza predicciones para nuevas muestras X.
        X_poly = self.caracteristicas_polinomiales(X)
        #Se calculan las características polinomiales de las nuevas muestras X.
        y_pred = np.dot(X_poly, self.coeficientes)
        #Se multiplican las características polinomiales por los coeficientes del modelo para obtener las predicciones.
        return y_pred
        #Se devuelve el array de predicciones.

    def ecuacion(self):
        #Se define un método llamado ecuacion, que devuelve la ecuación del modelo de regresión polinomial.
        ecuacion = "y = "
        #Se inicializa una cadena de texto para almacenar la ecuación.
        for i, coef in enumerate(self.coeficientes[::-1]):
            #Se itera sobre los coeficientes del modelo en orden inverso para construir la ecuación.
            ecuacion += f"{coef}x^{i}"
            #Se agrega cada término de la ecuación polinomial a la cadena.
            if i != len(self.coeficientes) - 1:
                ecuacion += " + "
            #Se agrega un signo más después de cada término, excepto el último.
        return ecuacion
        #Se devuelve la ecuación completa.

    def caracteristicas_polinomiales(self, X):
        #Se define un método interno caracteristicas_polinomiales, que calcula las características polinomiales de una matriz X.
        n, m = X.shape
        #Se obtienen las dimensiones de la matriz de entrada X.
        X_poly = np.ones((n, self.degree + 1))
        #Se inicializa una matriz de características polinomiales con unos.
        for d in range(1, self.degree + 1):
        #Se itera sobre los grados del polinomio.
            X_poly[:, d] = X[:, 0] ** d
        #Se calculan las características polinomiales para cada grado y se asignan a la matriz X_poly.
        return X_poly
        #Se devuelve la matriz de características polinomiales.

    def coeficientes_calculados(self, x_poly, y):
        #Se define un método interno _calculate_coefficients, que calcula los coeficientes del modelo de regresión polinomial.
        return np.linalg.inv(x_poly.T.dot(x_poly)).dot(x_poly.T).dot(y)
        #Se utiliza la fórmula de mínimos cuadrados para calcular los coeficientes del modelo y se devuelven.
        #calcula el inverso de la matriz
    def imprimir_ecuacion_de_regrecion(self):
        equation = "Equacion de Regresion: "
        for i, coef in enumerate(self.coeficientes[::-1]):
            #que itera sobre los coeficientes del modelo de regresión polinomial almacenados en self.coefficients. 
            #Utilizamos enumerate para obtener tanto el índice i como el valor del coeficiente coef. También usamos 
            #[::-1] para iterar sobre los coeficientes en orden inverso, ya que los coeficientes se almacenan en el 
            #orden opuesto al polinomio.
            equation += f"{coef} * x^{i}"
            #Dentro del bucle, añadimos cada término de la ecuación polinomial a la cadena equation. Usamos f-strings 
            #para formatear el coeficiente y el exponente de x en cada término.
            if i != len(self.coeficientes) - 1:
                equation += " + "
                #Esta condicion verifica si estamos en el último término de la ecuación. Si no lo estamos, añadimos 
                #un signo + después del término actual para indicar la suma de los términos de la ecuación.
        print(equation)
        #Finalmente, imprimimos la cadena equation, que contiene la ecuación completa de la regresión polinomial.

# Dataset
x = np.array([108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89]).reshape(-1, 1)
y = np.array([95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 86, 60, 63, 95, 61, 55, 56, 94, 93])

# Modelos lineal, cuadrático y cúbico
degrees = [1, 2, 3]
for degree in degrees:
    model = PolynomialRegression(degree)
    model.ajuste(x, y)
    print(f"Grado de Regresion Polinomial {degree} Ecuacion: {model.ecuacion()}")

    # Predicciones
    x_conocida = np.array([[108], [115], [106]])
    x_desconocida = np.array([[120], [125], [130]])
    print(f"Predicciones para X Conocida: {model.prediccion(x_conocida)}")
    print(f"Predicciones para X Desonocida: {model.prediccion(x_desconocida)}")
    
    model.imprimir_ecuacion_de_regrecion()