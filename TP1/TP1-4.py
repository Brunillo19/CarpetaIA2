import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cityblock
from matplotlib.patches import Rectangle
import heapq
import math

# Definir el almacén
almacen_dim = (11, 13)
estacion_carga = (5, 0)

# Definir posiciones de estanterías
estanterias = {i: (x, y) for i, (x, y) in enumerate([
    (1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (4, 3),
    (1, 6), (1, 7), (2, 6), (2, 7), (3, 6), (3, 7), (4, 6), (4, 7),
    (1, 10), (1, 11), (2, 10), (2, 11), (3, 10), (3, 11), (4, 10), (4, 11),
    (6, 2), (6, 3), (7, 2), (7, 3), (8, 2), (8, 3), (9, 2), (9, 3),
    (6, 6), (6, 7), (7, 6), (7, 7), (8, 6), (8, 7), (9, 6), (9, 7),
    (6, 10), (6, 11), (7, 10), (7, 11), (8, 10), (8, 11), (9, 10), (9, 11)
])}

# ----- PARTE 1: CARGAR Y ANALIZAR DATOS -----

# Cargar datos de órdenes
def cargar_ordenes(file_path = "ordenes.csv"):
    ordenes = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            productos = list(map(int, line.strip().split(",")))  # Convertir a enteros
            ordenes.append(productos)
    return ordenes

# Calcular frecuencia de pedidos
def analizar_frecuencia(ordenes):
    frecuencia_productos = {}
    for orden in ordenes:
        for producto in orden:
            if producto in frecuencia_productos:
                frecuencia_productos[producto] += 1
            else:
                frecuencia_productos[producto] = 1

    # Normalizar frecuencias
    max_frecuencia = max(frecuencia_productos.values())
    frecuencia_productos = {k: v / max_frecuencia for k, v in frecuencia_productos.items()}
    return frecuencia_productos

# ----- PARTE 2: ALGORITMO GENÉTICO -----

# Función de aptitud modificada para evaluar todas las órdenes
def calcular_fitness(individuo, ordenes, num_ordenes=50):
    """Evalúa la configuración minimizando el costo real de picking usando todas las órdenes disponibles."""
    configuracion_almacen = crear_mapa_almacen(individuo)
    mapa_binario = convertir_a_mapa_binario(configuracion_almacen)

    # Seleccionar órdenes para evaluar el picking (máximo 50 o todas si hay menos)
    ordenes_a_evaluar = ordenes[:min(num_ordenes, len(ordenes))]
    costo_total = 0

    for orden in ordenes_a_evaluar:
        posiciones = {'inicio': estacion_carga}
        productos_accesibles = []

        for producto in orden:
            if producto in individuo:
                punto_acceso = encontrar_punto_acceso(configuracion_almacen, producto)
                if punto_acceso:
                    posiciones[producto] = punto_acceso
                    productos_accesibles.append(producto)

        if productos_accesibles:
            # Optimizar ruta con temple simulado
            todos_los_caminos = precalcular_caminos(mapa_binario, posiciones)
            matriz_distancias, indices_puntos = calcular_matriz_distancias(todos_los_caminos)
            mejor_ruta, mejor_costo = recocido_simulado(productos_accesibles, posiciones, matriz_distancias, indices_puntos)
            costo_total += mejor_costo  # Sumar costos de picking para evaluar configuración

    # Retornar el costo total invertido para minimizar (menor costo = mejor fitness)
    return -costo_total if costo_total > 0 else 0

# Inicializar población
def inicializar_poblacion(tamano, productos):
    poblacion = []

    # Crear una configuración inicial basada en las estanterías
    configuracion_original = productos.copy()
    poblacion.append(configuracion_original)

    # Generar el resto de la población aleatoriamente
    for _ in range(tamano - 1):
        random.shuffle(productos)
        poblacion.append(productos.copy())

    return poblacion
# Selección por torneo
def seleccion_torneo(poblacion, fitness, k=5):
    participantes = random.sample(list(zip(poblacion, fitness)), k)
    return max(participantes, key=lambda x: x[1])[0]

# Cruce ordenado
def cruce_ox(padre1, padre2):
    n = len(padre1)
    corte1, corte2 = sorted(random.sample(range(n), 2))
    hijo = [-1] * n
    hijo[corte1:corte2] = padre1[corte1:corte2]
    restantes = [p for p in padre2 if p not in hijo]
    idx = 0
    for i in range(n):
        if hijo[i] == -1:
            hijo[i] = restantes[idx]
            idx += 1
    return hijo

# Mutación por intercambio
def mutacion(individuo, tasa=0.1):
    if random.random() < tasa:
        i, j = random.sample(range(len(individuo)), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo

# Algoritmo genético modificado para guardar el historial de evolución
def algoritmo_genetico(ordenes, tamano_poblacion=100, generaciones=50):
    productos = list(frecuencia_productos.keys())
    poblacion = inicializar_poblacion(tamano_poblacion, productos)
    mejor_individuo = None
    mejor_fitness = -float('inf')

    # Guardar el historial de evolución
    historial_fitness = []
    historial_configuraciones = []

    # Guardar la configuración inicial para comparación
    configuracion_inicial = poblacion[0].copy()

    for gen in range(generaciones):
        print(f"Generación {gen+1}/{generaciones}")

        # Evaluar fitness de todos los individuos
        fitness = [calcular_fitness(ind, ordenes) for ind in poblacion]

        # Actualizar el mejor individuo si se encuentra uno mejor
        gen_mejor_fitness = max(fitness)
        gen_mejor_individuo = poblacion[fitness.index(gen_mejor_fitness)]

        if gen_mejor_fitness > mejor_fitness:
            mejor_fitness = gen_mejor_fitness
            mejor_individuo = gen_mejor_individuo.copy()
            print(f"  Nuevo mejor fitness: {mejor_fitness}")

        # Guardar en historial
        historial_fitness.append(gen_mejor_fitness)
        historial_configuraciones.append(gen_mejor_individuo.copy())

        # Crear nueva población con selección, cruce y mutación
        nueva_poblacion = []
        for _ in range(tamano_poblacion // 2):
            padre1 = seleccion_torneo(poblacion, fitness)
            padre2 = seleccion_torneo(poblacion, fitness)
            hijo1 = mutacion(cruce_ox(padre1, padre2))
            hijo2 = mutacion(cruce_ox(padre2, padre1))
            nueva_poblacion.extend([hijo1, hijo2])

        poblacion = nueva_poblacion

    # Información de evolución
    evolucion_info = {
        "fitness_historial": historial_fitness,
        "configuraciones_historial": historial_configuraciones,
        "configuracion_inicial": configuracion_inicial,
        "configuracion_final": mejor_individuo
    }

    return mejor_individuo, evolucion_info

# ----- PARTE 3: ALGORITMO A* Y RUTA -----

# Convierte la configuración del almacén en una matriz binaria para navegación
def crear_mapa_almacen(configuracion):
    mapa = [['.' for _ in range(almacen_dim[1])] for _ in range(almacen_dim[0])]

    # Colocar la estación de carga
    mapa[estacion_carga[0]][estacion_carga[1]] = 'C'

    # Colocar productos en estanterías
    for idx, producto in enumerate(configuracion):
        if idx in estanterias:
            x, y = estanterias[idx]
            mapa[x][y] = str(producto)

    return mapa

# Convierte el mapa del almacén a una matriz binaria (0 = espacio libre, 1 = obstáculo)
def convertir_a_mapa_binario(almacen):
    binario = np.zeros((len(almacen), len(almacen[0])), dtype=np.int8)
    for y in range(len(almacen)):
        for x in range(len(almacen[0])):
            binario[y, x] = 0 if almacen[y][x] == '.' or almacen[y][x] == 'C' else 1
    return binario

# Distancia Manhattan entre dos puntos
def distancia_manhattan(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

# Algoritmo A* para encontrar el camino más corto
def a_estrella(mapa_binario, inicio, fin):
    filas, columnas = mapa_binario.shape
    movimientos = [(-1,0), (1,0), (0,-1), (0,1)]  # Movimientos posibles: arriba, abajo, izquierda, derecha
    conjunto_abierto = [(0, inicio)]  # Cola de prioridad para explorar nodos
    vino_de, puntaje_g, puntaje_f = {}, {inicio: 0}, {inicio: distancia_manhattan(inicio, fin)}

    while conjunto_abierto:
        _, actual = heapq.heappop(conjunto_abierto)  # Obtiene nodo con menor puntaje_f
        if actual == fin:  # Si llegamos al destino, reconstruimos el camino
            camino = [actual]
            while actual in vino_de:
                actual = vino_de[actual]
                camino.append(actual)
            return camino[::-1]  # Devuelve el camino invertido (del inicio al final)

        # Explora vecinos
        for dy, dx in movimientos:
            vecino = (actual[0]+dy, actual[1]+dx)
            if (0 <= vecino[0] < filas and 0 <= vecino[1] < columnas and
                mapa_binario[vecino[0], vecino[1]] == 0):  # Si es válido y transitable
                tentativo_g = puntaje_g[actual] + 1
                if vecino not in puntaje_g or tentativo_g < puntaje_g[vecino]:
                    vino_de[vecino] = actual
                    puntaje_g[vecino] = tentativo_g
                    puntaje_f[vecino] = tentativo_g + distancia_manhattan(vecino, fin)
                    heapq.heappush(conjunto_abierto, (puntaje_f[vecino], vecino))
    return []  # No se encontró camino

# Encuentra punto de acceso para un producto
def encontrar_punto_acceso(mapa_almacen, producto):
    filas, columnas = len(mapa_almacen), len(mapa_almacen[0])
    puntos_producto = [(y, x) for y in range(filas) for x in range(columnas)
                     if mapa_almacen[y][x] == str(producto)]

    for y, x in puntos_producto:
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Busca espacio libre alrededor
            ny, nx = y + dy, x + dx
            if 0 <= ny < filas and 0 <= nx < columnas and mapa_almacen[ny][nx] == '.':
                return (ny, nx)
    return None

# Precalcula todos los caminos posibles entre posiciones importantes
def precalcular_caminos(mapa_binario, posiciones):
    todos_los_caminos = {}
    puntos = list(posiciones.values())
    for i, inicio in enumerate(puntos):
        for j, fin in enumerate(puntos):
            if i != j:
                todos_los_caminos[(inicio, fin)] = a_estrella(mapa_binario, inicio, fin)
    return todos_los_caminos

# Calcula matriz de distancias entre todos los puntos
def calcular_matriz_distancias(todos_los_caminos):
    puntos = set()
    for inicio, fin in todos_los_caminos.keys():
        puntos.add(inicio)
        puntos.add(fin)

    puntos = list(puntos)
    n = len(puntos)
    matriz_distancias = np.full((n, n), np.inf)

    for i in range(n):
        for j in range(n):
            if i != j:
                camino = todos_los_caminos.get((puntos[i], puntos[j]), [])
                if camino:
                    matriz_distancias[i, j] = len(camino) - 1

    return matriz_distancias, {punto: i for i, punto in enumerate(puntos)}

# ----- PARTE 4: TEMPLE SIMULADO PARA OPTIMIZAR RUTAS -----

# Calcula el costo total de una ruta
def calcular_costo_ruta(ruta, posiciones, matriz_distancias, indices_puntos):
    costo = 0
    indice_actual = indices_puntos[posiciones['inicio']]  # Comienza desde el punto de inicio
    for item in ruta:
        indice_siguiente = indices_puntos[posiciones[item]]
        if matriz_distancias[indice_actual, indice_siguiente] == np.inf:
            return float('inf')
        costo += matriz_distancias[indice_actual, indice_siguiente]
        indice_actual = indice_siguiente

    # Volver al inicio
    indice_final = indices_puntos[posiciones['inicio']]
    costo += matriz_distancias[indice_actual, indice_final]

    return costo

# Genera variaciones de una ruta para explorar soluciones alternativas
def generar_vecinos(ruta, num=5):
    vecinos = []
    for _ in range(num):
        nueva_ruta = ruta.copy()
        operacion = random.randint(0, 2)

        if operacion == 0 and len(nueva_ruta) >= 2:  # Intercambiar dos elementos
            i, j = random.sample(range(len(nueva_ruta)), 2)
            nueva_ruta[i], nueva_ruta[j] = nueva_ruta[j], nueva_ruta[i]
        elif operacion == 1 and len(nueva_ruta) > 1:  # Mover un elemento a otra posición
            i, j = random.sample(range(len(nueva_ruta)), 2)
            item = nueva_ruta.pop(i)
            nueva_ruta.insert(j, item)
        elif operacion == 2 and len(nueva_ruta) > 2:  # Invertir una sección
            i, j = sorted(random.sample(range(len(nueva_ruta)), 2))
            nueva_ruta[i:j+1] = reversed(nueva_ruta[i:j+1])
        vecinos.append(nueva_ruta)
    return vecinos

# Algoritmo de recocido simulado
def recocido_simulado(ruta_inicial, posiciones, matriz_distancias, indices_puntos,
                      temperatura=1000, enfriamiento=0.6, iteraciones=100, max_estancamiento=20):
    mejor_ruta = ruta_actual = ruta_inicial.copy()
    mejor_costo = costo_actual = calcular_costo_ruta(ruta_actual, posiciones, matriz_distancias, indices_puntos)
    estancamiento = 0

    while temperatura > 0.1 and estancamiento < max_estancamiento:
        mejora = False

        for _ in range(iteraciones):
            vecinos = generar_vecinos(ruta_actual, num=5)

            for vecino in vecinos:
                nuevo_costo = calcular_costo_ruta(vecino, posiciones, matriz_distancias, indices_puntos)
                if nuevo_costo == float('inf'):
                    continue
                delta = nuevo_costo - costo_actual

                # Acepta solución si es mejor o con probabilidad basada en temperatura
                if delta < 0 or random.random() < math.exp(-delta / temperatura):
                    ruta_actual, costo_actual = vecino, nuevo_costo
                    if nuevo_costo < mejor_costo:
                        mejor_ruta, mejor_costo = vecino.copy(), nuevo_costo
                        mejora = True
                    if delta < 0:
                        break

        estancamiento = 0 if mejora else estancamiento + 1
        temperatura *= enfriamiento  # Enfría la temperatura gradualmente

    return mejor_ruta, mejor_costo

# ----- PARTE 5: INTEGRACIÓN Y VISUALIZACIÓN -----

# Optimizar una orden específica
def optimizar_orden(productos, configuracion, mapa_binario):
    # Convertir a mapa de almacén
    mapa_almacen = crear_mapa_almacen(configuracion)

    # Encontrar puntos de acceso para cada producto
    posiciones = {'inicio': estacion_carga}  # Posición inicial
    productos_accesibles = []

    for producto in productos:
        if producto in configuracion:
            punto_acceso = encontrar_punto_acceso(mapa_almacen, producto)
            if punto_acceso:
                posiciones[producto] = punto_acceso
                productos_accesibles.append(producto)

    if not productos_accesibles:
        return [], 0, "No se encontraron productos accesibles en esta orden"

    # Calcular caminos y matriz de distancias
    todos_los_caminos = precalcular_caminos(mapa_binario, posiciones)
    matriz_distancias, indices_puntos = calcular_matriz_distancias(todos_los_caminos)

    # Optimizar ruta con temple simulado
    mejor_ruta, mejor_costo = recocido_simulado(productos_accesibles, posiciones, matriz_distancias, indices_puntos)

    return mejor_ruta, mejor_costo, "Ruta optimizada exitosamente"

# Generar mapa de calor
def generar_mapa_calor(configuracion, titulo):
    matriz_almacen = np.zeros(almacen_dim)
    for idx, producto in enumerate(configuracion):
        if idx in estanterias:
            x, y = estanterias[idx]
            matriz_almacen[x, y] = frecuencia_productos.get(producto, 1)

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(matriz_almacen, cmap="YlOrRd", linewidths=0.5, linecolor='black', cbar_kws={'label': 'Frecuencia de pedidos'})
    plt.title(titulo)
    for idx, producto in enumerate(configuracion):
        if idx in estanterias:
            x, y = estanterias[idx]
            plt.text(y + 0.5, x + 0.5, f"{producto}", ha='center', va='center', fontsize=8, color="black", fontweight='bold')
    y, x = estacion_carga
    rect = Rectangle((x, y), 1, 1, fill=True, color='yellow', alpha=0.8)
    ax.add_patch(rect)
    plt.text(x + 0.5, y + 0.5, "Carga", ha='center', va='center', fontsize=8, color="black", fontweight='bold')
    plt.savefig(f"{titulo.replace(' ', '_')}.png")
    plt.close()

# Visualizar una ruta optimizada
def visualizar_ruta(configuracion, ruta, mapa_binario):
    mapa_almacen = crear_mapa_almacen(configuracion)

    # Encontrar puntos de acceso para cada producto
    posiciones = {'inicio': estacion_carga}  # Posición inicial
    for producto in ruta:
        punto_acceso = encontrar_punto_acceso(mapa_almacen, producto)
        if punto_acceso:
            posiciones[producto] = punto_acceso

    # Crear mapa para visualización
    matriz_ruta = np.zeros(almacen_dim)

    # Marcar posiciones de productos
    for idx, producto in enumerate(configuracion):
        if idx in estanterias:
            x, y = estanterias[idx]
            if producto in ruta:
                matriz_ruta[x, y] = 2  # Productos en la ruta
            else:
                matriz_ruta[x, y] = 1  # Otros productos

    # Marcar la estación de carga
    matriz_ruta[estacion_carga[0], estacion_carga[1]] = 3

    # Calcular el camino completo
    camino_completo = [estacion_carga]
    punto_actual = estacion_carga

    # Precalcular caminos
    todos_los_caminos = precalcular_caminos(mapa_binario, posiciones)

    for producto in ruta:
        if producto in posiciones:
            punto_destino = posiciones[producto]
            segmento = a_estrella(mapa_binario, punto_actual, punto_destino)
            if segmento:
                camino_completo.extend(segmento[1:])
                punto_actual = punto_destino

    # Volver a la estación de carga
    ultimo_segmento = a_estrella(mapa_binario, punto_actual, estacion_carga)
    if ultimo_segmento:
        camino_completo.extend(ultimo_segmento[1:])

    # Visualizar
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Colores para el mapa
    cmap = plt.cm.colors.ListedColormap(['white', 'lightgray', 'lightblue', 'green'])
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Mostrar mapa base
    plt.imshow(matriz_ruta, cmap=cmap, norm=norm)

    # Dibujar camino
    xs = [p[1] for p in camino_completo]
    ys = [p[0] for p in camino_completo]
    plt.plot(xs, ys, 'r-', linewidth=2, alpha=0.7)

    # Marcar puntos importantes
    plt.plot(estacion_carga[1], estacion_carga[0], 'go', markersize=10)
    plt.text(estacion_carga[1], estacion_carga[0], "Inicio/Fin", fontsize=8, ha='center', va='bottom')

    for i, producto in enumerate(ruta):
        if producto in posiciones:
            y, x = posiciones[producto]
            plt.plot(x, y, 'bo', markersize=8)
            plt.text(x, y, f"{i+1}. {producto}", fontsize=8, ha='center', va='bottom')

    # Agregar etiquetas a los productos
    for idx, producto in enumerate(configuracion):
        if idx in estanterias:
            x, y = estanterias[idx]
            plt.text(y, x, f"{producto}", ha='center', va='center', fontsize=8, color="black", fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.title(f"Ruta optimizada para {len(ruta)} productos")
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=[0.25, 1, 2, 3],
                 label='0: Pasillo, 1: Producto, 2: Producto en ruta, 3: Estación')
    plt.savefig(f"ruta_optimizada_{len(ruta)}_productos.png")
    plt.close()

# Visualizar evolución del fitness
def visualizar_evolucion(historial_fitness):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(historial_fitness) + 1), historial_fitness, 'b-', marker='o')
    plt.xlabel('Generación')
    plt.ylabel('Fitness (1 / (1 + costo_total))')
    plt.title('Evolución del Fitness durante el Algoritmo Genético')
    plt.grid(True)
    plt.savefig("evolucion_fitness.png")
    plt.close()

# Función para visualizar comparativa de configuraciones
def visualizar_comparativa(config_inicial, config_final):
    plt.figure(figsize=(20, 10))

    # Configuración inicial
    plt.subplot(1, 2, 1)
    generar_mapa_calor(config_inicial, "Configuración Inicial")

    # Configuración final
    plt.subplot(1, 2, 2)
    generar_mapa_calor(config_final, "Configuración Final (Optimizada)")

    plt.tight_layout()
    plt.savefig("comparativa_configuraciones.png")
    plt.close()

# Calcular costo total de picking para todas las órdenes
def calcular_costo_total(configuracion, ordenes, max_ordenes=50):
    configuracion_almacen = crear_mapa_almacen(configuracion)
    mapa_binario = convertir_a_mapa_binario(configuracion_almacen)

    ordenes_a_evaluar = ordenes[:min(max_ordenes, len(ordenes))]
    costo_total = 0

    for orden in ordenes_a_evaluar:
        ruta, costo, _ = optimizar_orden(orden, configuracion, mapa_binario)
        costo_total += costo

    return costo_total

# ----- PARTE 6: FUNCIÓN PRINCIPAL -----

def optimizar_almacen(file_path = "ordenes.csv", visualizar=True):
    print("1. Cargando datos y analizando frecuencias...")
    ordenes = cargar_ordenes(file_path)
    global frecuencia_productos
    frecuencia_productos = analizar_frecuencia(ordenes)

    print(f"Se cargaron {len(ordenes)} órdenes con {len(frecuencia_productos)} productos distintos")

    print("\n2. Optimizando ubicación de productos con algoritmo genético...")
    mejor_configuracion, evolucion_info = algoritmo_genetico(ordenes, tamano_poblacion=40, generaciones=20)

    # Usar la configuración inicial guardada
    configuracion_original = evolucion_info["configuracion_inicial"]

    # Crear mapa binario para navegación
    mapa_almacen = crear_mapa_almacen(mejor_configuracion)
    mapa_binario = convertir_a_mapa_binario(mapa_almacen)

    # Calcular costos totales para configuración original y optimizada
    print("\n3. Calculando costos totales para comparación...")
    costo_original = calcular_costo_total(configuracion_original, ordenes)
    costo_optimizado = calcular_costo_total(mejor_configuracion, ordenes)

    print(f"Costo total configuración original: {costo_original}")
    print(f"Costo total configuración optimizada: {costo_optimizado}")
    print(f"Mejora porcentual: {100 * (costo_original - costo_optimizado) / costo_original:.2f}%")

    if visualizar:
        print("\n4. Generando visualización comparativa de mapas de calor...")
        # Solo generar la visualización comparativa
        visualizar_comparativa_mapas_calor(configuracion_original, mejor_configuracion)




    return {
        "configuracion": mejor_configuracion,
        "mapa_binario": mapa_binario,

    }

# Nueva función para visualizar solo la comparativa de mapas de calor
def visualizar_comparativa_mapas_calor(config_inicial, config_final):
    plt.figure(figsize=(20, 8))

    # Configuración inicial
    plt.subplot(1, 2, 1)
    matriz_almacen_inicial = np.zeros(almacen_dim)
    for idx, producto in enumerate(config_inicial):
        if idx in estanterias:
            x, y = estanterias[idx]
            matriz_almacen_inicial[x, y] = frecuencia_productos.get(producto, 1)

    ax1 = plt.subplot(1, 2, 1)
    sns.heatmap(matriz_almacen_inicial, cmap="YlOrRd", linewidths=0.5, linecolor='black',
               cbar_kws={'label': 'Frecuencia de pedidos'}, ax=ax1)
    ax1.set_title("Configuración Inicial")

    # Añadir números de productos a cada estantería
    for idx, producto in enumerate(config_inicial):
        if idx in estanterias:
            x, y = estanterias[idx]
            ax1.text(y + 0.5, x + 0.5, f"{producto}", ha='center', va='center',
                    fontsize=8, color="black", fontweight='bold')

    # Marcar estación de carga
    y, x = estacion_carga
    rect1 = Rectangle((x, y), 1, 1, fill=True, color='yellow', alpha=0.8)
    ax1.add_patch(rect1)
    ax1.text(x + 0.5, y + 0.5, "Carga", ha='center', va='center', fontsize=8,
            color="black", fontweight='bold')

    # Configuración optimizada
    plt.subplot(1, 2, 2)
    matriz_almacen_final = np.zeros(almacen_dim)
    for idx, producto in enumerate(config_final):
        if idx in estanterias:
            x, y = estanterias[idx]
            matriz_almacen_final[x, y] = frecuencia_productos.get(producto, 1)

    ax2 = plt.subplot(1, 2, 2)
    sns.heatmap(matriz_almacen_final, cmap="YlOrRd", linewidths=0.5, linecolor='black',
               cbar_kws={'label': 'Frecuencia de pedidos'}, ax=ax2)
    ax2.set_title("Configuración Optimizada")

    # Añadir números de productos a cada estantería
    for idx, producto in enumerate(config_final):
        if idx in estanterias:
            x, y = estanterias[idx]
            ax2.text(y + 0.5, x + 0.5, f"{producto}", ha='center', va='center',
                    fontsize=8, color="black", fontweight='bold')

    # Marcar estación de carga
    y, x = estacion_carga
    rect2 = Rectangle((x, y), 1, 1, fill=True, color='yellow', alpha=0.8)
    ax2.add_patch(rect2)
    ax2.text(x + 0.5, y + 0.5, "Carga", ha='center', va='center', fontsize=8,
            color="black", fontweight='bold')

    plt.tight_layout()
    plt.savefig("comparativa_mapas_calor.png")
    plt.close()

# Ejemplo de uso
if __name__ == "__main__":
    resultado = optimizar_almacen()