# DataSet (hardcoded)
x = [23, 26, 30, 34, 43, 48, 52, 57, 58]
y = [651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]

# Calculando la media
media_x = sum(x) / 9
media_y = sum(y) / 9

# Calculo de beta1
numerador = sum((x - media_x) * (y - media_y) for x, y in zip(x, y))
denominador = sum((x - media_x) ** 2 for x in x)
beta1 = numerador / denominador

# Calculando beta0
beta0 = media_y - beta1 * media_x

# Ecuación de Regresión
print("Bienvenido a la Ecuación de Regresión, esta vez tenemos:")
print("ŷ = {} + {}x".format(beta0, beta1))

# Prediccion de  Y para la X dada
# x_input = 35 significa que estamos eligiendo un valor de entrada de 35 
# para X. Este valor se utiliza para predecir el valor correspondiente de 
# Y utilizando el modelo de regresión lineal que hemos ajustado utilizando 
# los datos de entrenamiento.

# # En otras palabras, x_input es el valor de la variable independiente (en 
# este caso, la inversión publicitaria) para el cual queremos predecir el 
# valor de la variable dependiente (en este caso, las ventas).

# # Entonces, cuando hacemos predicted_y = beta0 + beta1 * x_input, 
# estamos aplicando la ecuación de regresión lineal a nuestro valor 
# de entrada (x_input) para obtener la predicción correspondiente de 
# (predicted_y). Esto nos dará una estimación de las ventas que podríamos 
# esperar para una inversión publicitaria de 35 millones de euros, basada en 
# el modelo que hemos ajustado a nuestros datos de entrenamiento.
x_input = 35
y_predecida = beta0 + beta1 * x_input
print("Para X = {}, Predicción de Y = {}".format(x_input, y_predecida))

# Calculando coeficiente de correlacion
covarianza_xy = numerador / 9
# Desviaciones estandar de x y y
# Es la desviación estándar de la variable independiente (X) en el 
# conjunto de datos. La desviación estándar es una medida de dispersión 
# que indica cuánto varían los valores de una variable respecto a su media.

# En el contexto de la regresión lineal, la desviación estándar de X 
# se utiliza para calcular el coeficiente de correlación entre X e Y.
std_dev_x = (sum((x - media_x) ** 2 for x in x) / 9) ** 0.5
std_dev_y = (sum((y - media_y) ** 2 for y in y) / 9) ** 0.5
correlacion_coef = covarianza_xy / (std_dev_x * std_dev_y)
print("Coeficiente de correlación:", correlacion_coef)

# Calculando coeficiente de determinacion (R^2)
r_cuadrada = correlacion_coef ** 2
print("Coeficiente de determinación (R^2):", r_cuadrada)

# Predicciones con cinco nuevos puntos de datos
nuevo_x = [20, 25, 40, 45, 55]
for x in nuevo_x:
    y_pred = beta0 + beta1 * x
    print("Para X = {}, Predicción de Y = {}".format(x, y_pred))
