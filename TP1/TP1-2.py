import heapq
import pygame
import time

estanterias = {
    1: (1, 2), 2: (1, 3), 3: (2, 2), 4: (2, 3), 5: (3, 2), 6: (3, 3), 7: (4, 2), 8: (4, 3),
    9: (1, 6), 10: (1, 7), 11: (2, 6), 12: (2, 7), 13: (3, 6), 14: (3, 7), 15: (4, 6), 16: (4, 7),
    17: (1, 10), 18: (1, 11), 19: (2, 10), 20: (2, 11), 21: (3, 10), 22: (3, 11), 23: (4, 10), 24: (4, 11),
    25: (6, 2), 26: (6, 3), 27: (7, 2), 28: (7, 3), 29: (8, 2), 30: (8, 3), 31: (9, 2), 32: (9, 3),
    33: (6, 6), 34: (6, 7), 35: (7, 6), 36: (7, 7), 37: (8, 6), 38: (8, 7), 39: (9, 6), 40: (9, 7),
    41: (6, 10), 42: (6, 11), 43: (7, 10), 44: (7, 11), 45: (8, 10), 46: (8, 11), 47: (9, 10), 48: (9, 11),
}

def heuristica(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def obtener_vecinos(pos, almacen):
    x, y = pos
    vecinos = []
    movimientos = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in movimientos:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(almacen) and 0 <= ny < len(almacen[0]) and almacen[nx][ny] == 0:
            vecinos.append((nx, ny))
    return vecinos

def encontrar_meta_adyacente(almacen, numero_estanteria):
    """Busca la casilla accesible más cercana a la estantería dada por su número"""
    if numero_estanteria not in estanterias:
        raise ValueError(f"El número de estantería {numero_estanteria} no existe.")

    x, y = estanterias[numero_estanteria]

    if y > 0 and almacen[x][y - 1] == 0:
        return (x, y - 1)
    elif y < len(almacen[0]) - 1 and almacen[x][y + 1] == 0:
        return (x, y + 1)
    elif x > 0 and almacen[x - 1][y] == 0:
        return (x - 1, y)
    elif x < len(almacen) - 1 and almacen[x + 1][y] == 0:
        return (x + 1, y)

    return None  # No hay espacio adyacente

def busqueda_a_estrella(almacen, inicio, meta, posiciones_ocupadas=set()):
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
        for vecino in obtener_vecinos(actual, almacen):
            if vecino in posiciones_ocupadas:
                continue
            nuevo_costo = costo_acumulado[actual] + 1
            if vecino not in costo_acumulado or nuevo_costo < costo_acumulado[vecino]:
                costo_acumulado[vecino] = nuevo_costo
                prioridad = nuevo_costo + heuristica(vecino, meta)
                heapq.heappush(lista_abierta, (prioridad, vecino))
                de_donde_viene[vecino] = actual

    camino = []
    nodo = meta
    while nodo != inicio:
        camino.append(nodo)
        nodo = de_donde_viene.get(nodo, inicio)
    camino.append(inicio)
    camino.reverse()
    return camino

def ejecutar_simulacion(almacen, montacargas):
    posiciones_actuales = {i: montacarga[0] for i, montacarga in enumerate(montacargas)}
    metas = {i: encontrar_meta_adyacente(almacen, montacarga[1]) for i, montacarga in enumerate(montacargas)}
    caminos = {i: busqueda_a_estrella(almacen, posiciones_actuales[i], metas[i], set()) for i in posiciones_actuales}

    pygame.init()
    tam_celda = 50
    ancho = len(almacen[0]) * tam_celda
    alto = len(almacen) * tam_celda
    pantalla = pygame.display.set_mode((ancho, alto))
    pygame.display.set_caption("Almacén - Montacargas A*")

    colores = {1: (100, 100, 100), 0: (255, 255, 255)}
    colores_montacargas = [(0, 0, 255), (255, 0, 0)]
    fuente = pygame.font.Font(None, 30)

    corriendo = True
    while corriendo:
        pantalla.fill((255, 255, 255))

        for i in range(len(almacen)):
            for j in range(len(almacen[0])):
                color = colores[almacen[i][j]]
                pygame.draw.rect(pantalla, color, (j * tam_celda, i * tam_celda, tam_celda, tam_celda))
                pygame.draw.rect(pantalla, (0, 0, 0), (j * tam_celda, i * tam_celda, tam_celda, tam_celda), 1)

        for numero, (x, y) in estanterias.items():
            texto = fuente.render(str(numero), True, (255, 255, 255))
            pantalla.blit(texto, (y * tam_celda + 15, x * tam_celda + 10))

        posiciones_ocupadas = set()
        for i in caminos:
            if len(caminos[i]) > 1:
                siguiente_posicion = caminos[i][1]
                if siguiente_posicion in posiciones_ocupadas:
                    caminos[i] = busqueda_a_estrella(almacen, posiciones_actuales[i], metas[i], posiciones_ocupadas)
                else:
                    posiciones_actuales[i] = siguiente_posicion
                    caminos[i] = caminos[i][1:]

            posiciones_ocupadas.add(posiciones_actuales[i])
            pygame.draw.rect(pantalla, colores_montacargas[i % len(colores_montacargas)],
                             (posiciones_actuales[i][1] * tam_celda, posiciones_actuales[i][0] * tam_celda, tam_celda, tam_celda))

        pygame.display.flip()
        time.sleep(0.5)

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False

    pygame.quit()


almacen = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

montacargas = [
    ((5, 1), 16),
    ((5, 12), 15)
]

ejecutar_simulacion(almacen, montacargas)
