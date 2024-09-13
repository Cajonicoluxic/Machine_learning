
# README

Este repositorio contiene varios notebooks de Jupyter, cada uno enfocado en distintos problemas de análisis de datos, machine learning y procesamiento. A continuación, se ofrece una descripción detallada de cada uno de los notebooks incluidos:

### 1. **Ejercicio3 (1).ipynb**

Este notebook está enfocado en hacer recomendaciones de películas utilizando el dataset de películas y calificaciones.

#### Explicación paso a paso:
1. **Cargar los datos**:
   ```python
   movies = pd.read_csv('movie.csv')
   ratings = pd.read_csv('rating.csv')
   ```
   Se cargan dos archivos CSV: uno con las películas (`movies`) y otro con las calificaciones (`ratings`).

2. **Filtrar películas de horror**:
   ```python
   horror_movies = movies[movies['genres'].str.contains('Horror')]
   ```
   Se filtran las películas que pertenecen al género de terror.

3. **Unir las películas de horror con las calificaciones**:
   ```python
   horror_ratings = pd.merge(horror_movies, ratings, on='movieId')
   ```
   Se unen los datos de las películas de horror con las calificaciones usando el identificador de película (`movieId`).

4. **Calcular las calificaciones promedio**:
   ```python
   horror_avg_ratings = horror_ratings.groupby('title').agg({'rating': 'mean'}).reset_index()
   ```
   Se agrupan las películas por su título y se calcula el promedio de calificaciones de cada película de horror.

5. **Ordenar las películas por calificación**:
   ```python
   horror_top_movies = horror_avg_ratings.sort_values(by='rating', ascending=False)
   print(horror_top_movies.head(10))
   ```
   Se ordenan las películas de terror por sus calificaciones promedio de mayor a menor, mostrando las 10 mejores.

6. **Recomendación de películas similares a "Toy Story"**:
   - Se encuentra el `movieId` de "Toy Story":
     ```python
     toy_story_id = movies[movies['title'].str.contains('Toy Story')]['movieId'].values[0]
     ```
   - Se extraen las calificaciones de los usuarios que han visto "Toy Story":
     ```python
     toy_story_ratings = ratings[ratings['movieId'] == toy_story_id]
     users_who_rated_toy_story = toy_story_ratings['userId'].unique()
     ```
   - Se generan recomendaciones de películas basadas en los usuarios que vieron "Toy Story":
     ```python
     user_ratings = ratings[ratings['userId'].isin(users_who_rated_toy_story)]
     movie_recommendations = user_ratings.groupby('movieId').agg({'rating': 'mean', 'userId': 'count'}).reset_index()
     movie_recommendations = pd.merge(movie_recommendations, movies, on='movieId')
     movie_recommendations = movie_recommendations[movie_recommendations['movieId'] != toy_story_id]
     movie_recommendations = movie_recommendations.sort_values(by=['rating', 'userId'], ascending=False)
     print(movie_recommendations.head(10))
     ```

#### Resumen:
El notebook carga datos de películas y calificaciones, y luego filtra las películas de terror para mostrar las más populares. Además, recomienda películas basadas en los usuarios que vieron "Toy Story".

---

### 2. **Ejercicio4.ipynb**

Este notebook parece centrarse en la detección de fraudes y la preparación de datos para un modelo de machine learning.

#### Explicación paso a paso:
1. **Cargar el dataset**:
   ```python
   url = 'reemplaza por el nombre del csv.csv'
   data = pd.read_csv(url)
   print(data.head())
   ```
   Se carga un archivo CSV (el nombre del archivo debe ser reemplazado en el código) y se visualizan los primeros registros.

2. **Preparación de los datos**:
   - Separar características y variable objetivo (fraude o no):
     ```python
     X = data.drop('Class', axis=1)
     y = data['Class']
     ```
   - Dividir el dataset en conjuntos de entrenamiento y prueba:
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     ```

3. **Normalización de los datos**:
   Se normalizan las características utilizando `StandardScaler`:
   ```python
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   ```

4. **Entrenamiento de un modelo de clasificación**:
   - Se entrena un modelo de regresión logística o cualquier otro modelo que se desee implementar (el código solo indica la preparación de los datos hasta el punto de entrenamiento, pero no incluye el modelo).

#### Resumen:
Este notebook carga un conjunto de datos y lo prepara para la construcción de un modelo de machine learning. Se divide el dataset en entrenamiento y prueba, y se normalizan las características.

---

### 3. **pregunta2.ipynb**

Este notebook parece estar centrado en el procesamiento de un dataset de spam.

#### Explicación paso a paso:
1. **Cargar los datos**:
   ```python
   spambase = pd.read_csv('spambase.csv', header=None)
   ```
   Se carga el dataset de spam sin nombres de columnas, ya que no tiene encabezado.

2. **Explorar el dataset**:
   ```python
   print(spambase.head())
   ```
   Se muestra una vista previa de los primeros registros del dataset.

3. **Preprocesamiento de datos**:
   - Se realiza el preprocesamiento de los datos, que puede incluir el manejo de valores faltantes o la normalización.

4. **Entrenamiento de un modelo**:
   - A partir de aquí, se entrena un modelo de clasificación para detectar si un correo es spam o no, aunque el código específico del modelo no está incluido en la parte extraída.

#### Resumen:
El notebook está enfocado en la detección de spam utilizando un dataset clásico, pero el modelo específico que se implementa no está claro en este fragmento.

---

### 4. **pregunta1.ipynb**

Este notebook trata sobre la predicción del valor de una casa utilizando un modelo de regresión.

#### Explicación paso a paso:
1. **Preprocesamiento de los datos**:
   - Se cargan los datos, se manejan valores faltantes, y se transforman variables categóricas en numéricas.

2. **Entrenamiento de un modelo de regresión lineal**:
   - Los datos se dividen en características (`X`) y la variable objetivo (`y`).
   - Se entrena un modelo de regresión lineal para predecir el valor de las casas.

#### Resumen:
Este notebook se centra en la predicción del valor de las casas utilizando regresión lineal, basado en el preprocesamiento de datos de viviendas.

---

### 5. **extra_perceptron.ipynb**

Este notebook implementa un Perceptrón utilizando el conjunto de datos **Iris**.

#### Explicación paso a paso:
1. **Cargar los datos**:
   ```python
   from sklearn.datasets import load_iris
   iris = load_iris()
   ```
   Se carga el famoso dataset Iris de `sklearn`.

2. **Dividir el dataset**:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   ```

3. **Entrenar el Perceptrón**:
   ```python
   perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
   perceptron.fit(X_train, y_train)
   ```

4. **Evaluar el modelo**:
   ```python
   y_pred = perceptron.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Exactitud del Perceptrón: {accuracy * 100:.2f}%")
   ```

5. **Visualización de la frontera de decisión**:
   Se genera una gráfica que muestra cómo el Perceptrón separa las clases.

#### Resumen:
Este notebook implementa un modelo de Perceptrón para clasificar flores del dataset Iris y visualiza la frontera de decisión del modelo.

---

Si necesitas que profundice más en algún aspecto, por favor, avísame.
