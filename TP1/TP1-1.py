import heapq
import pygame

# Diccionario con las coordenadas de cada estantería
estanterias = {
    1: (1, 2), 2: (1, 3), 3: (2, 2), 4: (2, 3), 5: (3, 2), 6: (3, 3), 7: (4, 2), 8: (4, 3),
    9: (1, 6), 10: (1, 7), 11: (2, 6), 12: (2, 7), 13: (3, 6), 14: (3, 7), 15: (4, 6), 16: (4, 7),
    17: (1, 10), 18: (1, 11), 19: (2, 10), 20: (2, 11), 21: (3, 10), 22: (3, 11), 23: (4, 10), 24: (4, 11),
    25: (6, 2), 26: (6, 3), 27: (7, 2), 28: (7, 3), 29: (8, 2), 30: (8, 3), 31: (9, 2), 32: (9, 3),
    33: (6, 6), 34: (6, 7), 35: (7, 6), 36: (7, 7), 37: (8, 6), 38: (8, 7), 39: (9, 6), 40: (9, 7),
    41: (6, 10), 42: (6, 11), 43: (7, 10), 44: (7, 11), 45: (8, 10), 46: (8, 11), 47: (9, 10), 48: (9, 11),
}

# Función para calcular la heurística (distancia de Manhattan)
def heuristica(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Función para obtener los vecinos accesibles desde una posición dada
def obtener_vecinos(pos):
    x, y = pos
    vecinos = []
    movimientos = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for dx, dy in movimientos:
        nx, ny = x + dx, y + dy
        if (nx, ny) not in estanterias.values():  # Verificar que no sea una estantería
            vecinos.append((nx, ny))

    return vecinos

# Función para encontrar la meta adyacente a una estantería (solo por los costados)
def encontrar_meta_adyacente(numero_estanteria):
    """Encuentra la casilla accesible más cercana horizontalmente (izquierda o derecha) a la estantería."""
    if numero_estanteria not in estanterias:
        raise ValueError(f"La estantería {numero_estanteria} no existe.")

    x, y = estanterias[numero_estanteria]

    # Intentar moverse solo horizontalmente (izquierda o derecha)
    if y > 0 and (x, y - 1) not in estanterias.values():
        return (x, y - 1)  # A la izquierda
    elif y < 12 and (x, y + 1) not in estanterias.values():
        return (x, y + 1)  # A la derecha

    return None  # No encontró espacio accesible en los costados

# Algoritmo A* para encontrar el camino óptimo
def busqueda_a_estrella(inicio, meta):
    if meta is None:
        return []  # Si no hay meta válida, no hay camino

    lista_abierta = []
    heapq.heappush(lista_abierta, (0, inicio))
    de_donde_viene = {}
    costo_acumulado = {inicio: 0}

    while lista_abierta:
        _, actual = heapq.heappop(lista_abierta)
        if actual == meta:
            break

        for vecino in obtener_vecinos(actual):
            nuevo_costo = costo_acumulado[actual] + 1
            if vecino not in costo_acumulado or nuevo_costo < costo_acumulado[vecino]:
                costo_acumulado[vecino] = nuevo_costo
                prioridad = nuevo_costo + heuristica(vecino, meta)
                heapq.heappush(lista_abierta, (prioridad, vecino))
                de_donde_viene[vecino] = actual

    # Reconstrucción del camino
    camino = []
    nodo = meta
    while nodo != inicio:
        camino.append(nodo)
        nodo = de_donde_viene.get(nodo, inicio)
    camino.append(inicio)
    camino.reverse()
    return camino

# Función para dibujar el almacén con pygame
def dibujar_almacen(pantalla, camino, inicio, meta):
    tam_celda = 50
    font = pygame.font.Font(None, 24)

    color_estanteria = (100, 100, 100)  # Gris
    color_pasillo = (255, 255, 255)     # Blanco
    color_borde = (0, 0, 0)             # Negro

    pantalla.fill(color_pasillo)

    for i in range(11):  # Filas
        for j in range(13):  # Columnas
            pos = (i, j)
            if pos in estanterias.values():
                color = color_estanteria  # Es una estantería
            else:
                color = color_pasillo  # Es un pasillo

            pygame.draw.rect(pantalla, color, (j * tam_celda, i * tam_celda, tam_celda, tam_celda))
            pygame.draw.rect(pantalla, color_borde, (j * tam_celda, i * tam_celda, tam_celda, tam_celda), 1)

    for numero, (x, y) in estanterias.items():
        texto = font.render(str(numero), True, (255, 255, 255))
        pantalla.blit(texto, (y * tam_celda + 15, x * tam_celda + 15))

    for nodo in camino:
        pygame.draw.rect(pantalla, (0, 0, 255), (nodo[1] * tam_celda, nodo[0] * tam_celda, tam_celda, tam_celda))

    pygame.draw.rect(pantalla, (0, 255, 0), (inicio[1] * tam_celda, inicio[0] * tam_celda, tam_celda, tam_celda))  # Inicio (Verde)
    pygame.draw.rect(pantalla, (255, 0, 0), (meta[1] * tam_celda, meta[0] * tam_celda, tam_celda, tam_celda))  # Meta (Rojo)

    pygame.display.flip()

# --- Main ---
print("\nSeleccione la opción para la posición inicial:")
print("1) Elegir la posición de inicio manualmente.")
print("2) Iniciar en la estación de carga.")
print("3) Salir.")

opcion = input("Ingrese 1, 2 o 3: ")

if opcion == "3":
    print("Saliendo del programa...")
    exit()

if opcion == "1":
    while True:
        fila_inicio = int(input("Fila inicio: "))
        columna_inicio = int(input("Columna inicio: "))

        if (fila_inicio, columna_inicio) not in estanterias.values():
            posicion_inicial = (fila_inicio, columna_inicio)
            break
        else:
            print("Error: No puedes iniciar en una estantería. Elige otra posición.")
else:
    posicion_inicial = (5, 0)  # Estación de carga

numero_estanteria = int(input("Ingrese el número de estantería (1-48): "))
meta_adyacente = encontrar_meta_adyacente(numero_estanteria)
camino = busqueda_a_estrella(posicion_inicial, meta_adyacente)

pygame.init()
pantalla = pygame.display.set_mode((13 * 50, 11 * 50))
pygame.display.set_caption("Almacén - Montacargas A*")

dibujar_almacen(pantalla, camino, posicion_inicial, meta_adyacente)

while True:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            exit()
