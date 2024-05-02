# TripleTen
### Projects I developed during my experience at TripleTen

#### Análisis del riesgo de incumplimiento de los prestatarios

En este proyecto, analizaremos un conjunto de datos sobre la solvencia crediticia de los clientes del banco. El análisis será utilizado para crear una puntuación de crédito para potenciales clientes, mejorando así el proceso de evaluación de crédito. Nuestro objetivo es identificar posibles conexiones entre las variables y el cumplimiento de pagos. El enfoque se divide en dos etapas principales: el preprocesamiento de datos para asegurar su calidad y la respuesta a preguntas claves sobre la relación entre tener hijos, el estado civil, el nivel de ingresos y el propósito del préstamo con el pago a tiempo.

#### ¿Qué vende un coche?

Análisis de Precio en el mercado de autos:

- Identificar factores clave para el precio segun los modelos más populares
- Relacionar precio con edad, millaje, condición y tipo de transmisión y color
- Trazar gráficos de caja y bigotes para las variables categóricas y gráficos de dispersión para el resto

#### ¿Cuál es un mejor plan?
Proyecto para el operador de telecomunicaciones Megaline. La empresa ofrece a sus clientes dos tarifas de prepago, Surf y Ultimate. El departamento comercial quiere saber cuál de los planes genera más ingresos para poder ajustar el presupuesto de publicidad.


Realizamos un análisis preliminar de las tarifas basado en una selección de clientes relativamente pequeña. Analizando los datos de 500 clientes de Megaline: quiénes son los clientes, de dónde son, qué tarifa usan, así como la cantidad de llamadas que hicieron y los mensajes de texto que enviaron en 2018. Analizamos el comportamiento de los clientes y determinamos qué tarifa de prepago genera más ingresos.

#### ¿Qué vende un juego?

Este proyecto implica el análisis de datos de una tienda en línea llamada ICE que vende videojuegos en todo el mundo. El objetivo principal es identificar patrones que determinen el éxito de un juego y planificar campañas publicitarias en base a esos patrones.

#### Análisis del impacto del clima en los viajes de fin de semana en Chicago

Prueba de Hipótesis:

La duración promedio de los viajes desde el Loop hasta el Aeropuerto Internacional O'Hare cambia los sábados lluviosos

#### Priorización de Hipótesis y Análisis de Test A/B

En colaboración con el __departamento de marketing__, se han recopilado __nueve hipótesis__ para __aumentar los ingresos__.

__Priorización de Hipótesis:__

- Aplicar el __framework ICE__ para priorizar hipótesis y ordenarlas de forma descendente
- Aplicar el __framework RICE__ para priorizar hipótesis y ordenarlas de forma descendente
- Explicar cómo cambia la __priorización__ al usar RICE en lugar de ICE, proporcionando una explicación de los cambios

__Análisis de Test A/B:__

- Representar gráficamente el __ingreso acumulado__ por grupo
- Representar gráficamente el __tamaño de pedido promedio__ acumulado por grupo
- Representar gráficamente la __diferencia relativa en el tamaño de pedido promedio__ entre el grupo B y el grupo A 
- Calcular la __tasa de conversión__ de cada grupo y representar gráficamente las __tasas diarias__
- Realizar un gráfico de dispersión del __número de pedidos por usuario__
- Calcular los __percentiles__ 95 y 99 para el número de pedidos por usuario y definir el __punto de anomalía__
- Realizar un gráfico de dispersión de los __precios de los pedidos__
- Calcular los __percentiles__ 95 y 99 de los precios de los pedidos y definir el __punto de anomalía__

__Utilizando datos en bruto:__
- Determinar la significancia estadística de las __diferencias en la conversión__ entre grupos
- Determinar la significancia estadística de las __diferencias en el tamaño promedio__ de pedido entre grupos 

__Utilizando datos filtrados:__
- Determinar la significancia estadística de las __diferencias en la conversión__ entre grupos
- Determinar la significancia estadística de las __diferencias en el tamaño promedio de pedido__ entre grupos


__Tomar una decisión basada en los resultados de la prueba, entre tres opciones:__ 
- Considerar a uno de los grupos como líder
- Concluir que no hay diferencia entre los grupos
- Continuar la prueba

#### El pequeño café regentado por robots en Los Ángeles

__Descripción del Proyecto:__

- Cómo manejar un pequeño café __regentado por robots__ en Los Ángeles
- El proyecto es prometedor pero caro, por lo que se buscamos __atraer inversores__

__Preguntas clave:__
- Interés en conocer las __condiciones actuales del mercado__
- ¿Se mantendrá el __éxito__ cuando la novedad de los camareros robots desaparezca?
- Investigar las proporciones de los __diferentes tipos de establecimientos__ y graficar los resultados
- Investigar las proporciones de __establecimientos que pertenecen a una cadena__ y los que no, y graficarlos
- __Identificar__ qué tipo de establecimiento es comúnmente una cadena
- Las cadenas suelen tener __muchos establecimientos con pocos asientos__ o __pocos establecimientos con muchos asientos__?
- Calcular el __promedio de número de asientos__ para cada tipo de restaurante y graficarlos
- Separar los datos de las direcciones de la columna `address` en una columna aparte
- Graficar las __diez mejores calles__ por número de restaurantes
- Encontrar el número de calles que tienen __solo un restaurante__
- Analizar la __distribución del número de asientos__ en calles con muchos restaurantes

#### Estudio A/B: Evaluación del nuevo sistema de recomendaciones

__Descripción técnica__

- __Nombre de la prueba__: `recommender_system_test`
- __Grupos__: А (_control_), B (_nuevo embudo de pago_)
- __Launch date__: _2020-12-07_
- Fecha en la que dejaron de aceptar nuevos usuarios: _2020-12-21_
- __Fecha de finalización__: _2021-01-01_
- __Audiencia__: __15%__ de los nuevos usuarios de la __región de la UE__
- __Propósito de la prueba__: probar cambios relacionados con la introducción de un __sistema de recomendaciones mejorado__
- __Resultado esperado__: dentro de los 14 días posteriores a la inscripción, los usuarios mostrarán una __mejor conversión__ en visitas de la página del producto (el evento `product_page`), instancias de agregar artículos al carrito de compras (`product_cart`) y compras (`purchase`) de los dos grupos
- En cada etapa del embudo `product_page` → `product_cart` → `purchase`, habrá al menos un __10% de aumento__
- __Número previsto de participantes__ de la prueba: _6k_

__Descargar__ los datos de la prueba y __comprobar__ si se ha realizado correctamente; __analizar__ los resultados.

#### Estudio A/B: Evaluación del nuevo sistema de recomendaciones

__Descripción técnica__

- __Nombre de la prueba__: `recommender_system_test`
- __Grupos__: А (_control_), B (_nuevo embudo de pago_)
- __Launch date__: _2020-12-07_
- Fecha en la que dejaron de aceptar nuevos usuarios: _2020-12-21_
- __Fecha de finalización__: _2021-01-01_
- __Audiencia__: __15%__ de los nuevos usuarios de la __región de la UE__
- __Propósito de la prueba__: probar cambios relacionados con la introducción de un __sistema de recomendaciones mejorado__
- __Resultado esperado__: dentro de los 14 días posteriores a la inscripción, los usuarios mostrarán una __mejor conversión__ en visitas de la página del producto (el evento `product_page`), instancias de agregar artículos al carrito de compras (`product_cart`) y compras (`purchase`) de los dos grupos
- En cada etapa del embudo `product_page` → `product_cart` → `purchase`, habrá al menos un __10% de aumento__
- __Número previsto de participantes__ de la prueba: _6k_

__Descargar__ los datos de la prueba y __comprobar__ si se ha realizado correctamente; __analizar__ los resultados.

#### Data-driven insights para la experiencia del lector

En respuesta al cambio global causado por la pandemia, surge nuestro proyecto, una iniciativa centrada en la lectura y respaldada por una base de datos rica en detalles sobre libros, autores y opiniones de lectores. En este nuevo panorama, donde más personas optan por quedarse en casa, nuestra propuesta busca transformar la experiencia de la lectura. Exploraremos cómo los datos pueden no solo contar historias, sino también crear conexiones significativas entre lectores, libros y escritores. ¡Bienvenido a una nueva era en la que la lectura va más allá de las páginas impresas!

__Exploración de Libros Post-2000:__

Analizaremos la cantidad de libros publicados después del 1 de enero de 2000 para comprender la relevancia y la frescura de la colección. Esto permitirá ofrecer a los usuarios una selección actualizada y acorde con las preferencias contemporáneas.

__Análisis Integral de Reseñas:__

Investigaremos el número de reseñas de usuarios para cada libro, junto con la calificación promedio asociada. Este análisis revelará patrones de preferencias y nos ayudará a entender la respuesta de la comunidad de lectores a cada obra.

__Identificación de la Editorial Destacada:__

Enfocaremos nuestro análisis en identificar la editorial que ha publicado el mayor número de libros con más de 50 páginas. Esto nos permitirá destacar la diversidad y profundidad editorial, excluyendo posibles publicaciones de menor envergadura.

__Reconocimiento del Autor Destacado:__

Identificaremos al autor con la calificación promedio más alta, limitándonos a libros que hayan recibido al menos 50 calificaciones. Este hallazgo destacará la excelencia literaria y proporcionará a los usuarios una referencia de calidad en la elección de sus lecturas.

__Promedio de Reseñas de Texto para Lectores Comprometidos:__

Exploraremos el número promedio de reseñas de texto proporcionadas por usuarios que han calificado más de 50 libros. Este análisis revelará el nivel de compromiso y la profundidad de análisis de los lectores más dedicados, ofreciendo información valiosa sobre la calidad de las opiniones.
Estas investigaciones se orientan a proporcionar una visión completa y detallada del contenido de la base de datos, permitiéndonos desarrollar un producto que responda de manera efectiva a las necesidades y expectativas de los amantes de la lectura. La combinación de datos sobre publicaciones, opiniones y preferencias literarias será clave para ofrecer una experiencia única y enriquecedora.

#### Segmentación de Usuarios en Aplicación Móvil

__Estudio del Embudo de Eventos:__

- Identificar eventos en un __embudo secuencial__
- Observar la __frecuencia de ocurrencia__ de cada evento
- Encontrar la cantidad de __usuarios para cada acción__ y calcular la __proporción__

__Análisis del Embudo de Eventos:__

- Utilizar el embudo para encontrar la __proporción de usuarios__ que pasan de una etapa a la siguiente
- Identificar la etapa en la que __se pierden más usuarios__
- Calcular el porcentaje de __usuarios que llegaron hasta el evento__ `contacts_show`

__Segmentación de Usuarios:__

- Agrupar usuarios en __segmentos__ basados en las acciones realizadas
- Utilizar __métricas__ como _tasa de retención_, _tiempo dedicado_ y _frecuencia de eventos_ para __caracterizar los segmentos__
- Utilizar __técnicas de clustering__ (como k-means) para evaluar la interpretabilidad de los segmentos resultantes

__Hipótesis y Prueba Estadística:__

- Existe una diferencia en la conversión en `contacts_show` entre usuarios que descargaron de __bing__ y __Google__

__Prueba Estadística:__

- Seleccionar la __prueba estadística__ adecuada (p. ej., t-test).
- __Dividir los datos__ en dos grupos: usuarios de __bing__ y usuarios de __Google__
- Calcular la __conversión__ en `contacts_show` __para cada grupo__
- Realizar la __prueba estadística__ para validar o refutar la hipótesis

#### Comportamiento del usuario para la aplicación de la empresa

- Preparar los datos para el análisis
- Estudiar y comprobar los datos
- ¿Cuántos eventos hay en los registros?
- ¿Cuántos usuarios hay en los registros?
- ¿Cuál es el promedio de eventos por usuario?
- ¿Qué periodo de tiempo cubren los datos? 
- Trazar un histograma por fecha y hora
- Encontrar el momento en el que los datos comienzan a estar completos
- Excluir la sección anterior
- Estudiar el embudo de eventos
- Observar la frecuencia de ocurrencia de cada evento
- Encontrar la cantidad de usuarios que realizaron cada acción y calcular la proporción
- Utiliza el embudo de eventos para encontrar la proporción de usuarios que pasan de una etapa a la siguiente
- Identificar en qué etapa se pierden más usuarios
- Calcular el porcentaje de usuarios que completan todo el viaje desde el primer evento hasta el pago

#### Análisis y Estrategia de Retención de Clientes para Model Fitness

__Modelo de Predicción de Cancelación:__

- Construir un modelo de __clasificación binaria__ para predecir cancelación
- Dividir los datos en conjuntos de __entrenamiento__ y __validación__
- Entrenar modelos de __regresión logística__ y __bosque aleatorio__
- Evaluar la __exactitud__, __precisión__ y __recall__ para comparar modelos
- Crear __clústeres de usuarios__
- Entrenar __modelo K-means__ para predecir clústeres de clientes
- __Estandarizar__ los datos y usar `linkage()` para dendrograma
- Analizar valores medios y distribuciones de __características por clúster__
- Calcular la __tasa de cancelación__ por clúster

__Conclusiones y Recomendaciones:__

- __Identificar__ grupos de clientes propensos a la cancelación
- Sugerir __medidas de retención__ basadas en los hallazgos del modelo y clústeres
- Proporcionar recomendaciones de marketing específicas para __mejorar la retención de clientes__
