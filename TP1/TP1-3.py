import heapq
import pygame
import random
import math

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

def encontrar_meta_adyacente(almacen, estanteria):
    x, y = estanteria
    if y > 0 and almacen[x][y - 1] == 0:
        return (x, y - 1)
    elif y < len(almacen[0]) - 1 and almacen[x][y + 1] == 0:
        return (x, y + 1)
    return None

def busqueda_a_estrella(almacen, inicio, meta):
    lista_abierta = []
    heapq.heappush(lista_abierta, (0, inicio))
    de_donde_viene = {}
    costo_acumulado = {inicio: 0}

    while lista_abierta:
        _, actual = heapq.heappop(lista_abierta)
        if actual == meta:
            break
        for vecino in obtener_vecinos(actual, almacen):
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
    return camino, len(camino) - 1

def calcular_costo_total(almacen, orden, inicio):
    costo_total = 0
    actual = inicio
    for estanteria in orden:
        meta = encontrar_meta_adyacente(almacen, estanteria)
        _, costo = busqueda_a_estrella(almacen, actual, meta)
        costo_total += costo
        actual = meta
    return costo_total

def temple_simulado(almacen, inicio, estanterias, T_inicial=1000, T_min=1, alfa=0.995):
    orden_actual = estanterias[:]
    random.shuffle(orden_actual)
    mejor_orden = orden_actual[:]
    mejor_costo = calcular_costo_total(almacen, mejor_orden, inicio)
    T = T_inicial
    iteraciones = 0  # Contador de iteraciones

    while T > T_min:
        iteraciones += 1  # Aumenta el contador de iteraciones
        nuevo_orden = orden_actual[:]
        i, j = random.sample(range(len(nuevo_orden)), 2)
        nuevo_orden[i], nuevo_orden[j] = nuevo_orden[j], nuevo_orden[i]
        nuevo_costo = calcular_costo_total(almacen, nuevo_orden, inicio)
        
        if nuevo_costo < mejor_costo or random.random() < math.exp((mejor_costo - nuevo_costo) / T):
            orden_actual = nuevo_orden[:]
            if nuevo_costo < mejor_costo:
                mejor_orden = nuevo_orden[:]
                mejor_costo = nuevo_costo
        
        T *= alfa
    
    return mejor_orden, mejor_costo, iteraciones
def dibujar_ruta(almacen, inicio, orden, costo_total, estanterias):
    pygame.init()
    tam_celda = 50
    colores_camino = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 165, 0)]
    pantalla = pygame.display.set_mode((len(almacen[0]) * tam_celda, len(almacen) * tam_celda))
    pygame.display.set_caption(f"Ruta Óptima - Costo: {costo_total}")
    pantalla.fill((255, 255, 255))
    font = pygame.font.Font(None, 24)
    
    for i in range(len(almacen)):
        for j in range(len(almacen[0])):
            color = (100, 100, 100) if almacen[i][j] == 1 else (255, 255, 255)
            pygame.draw.rect(pantalla, color, (j * tam_celda, i * tam_celda, tam_celda, tam_celda))
            pygame.draw.rect(pantalla, (0, 0, 0), (j * tam_celda, i * tam_celda, tam_celda, tam_celda), 1)
            
            for num, (x, y) in estanterias.items():
                if (i, j) == (x, y):
                    texto = font.render(str(num), True, (255, 255, 255))
                    pantalla.blit(texto, (j * tam_celda + 15, i * tam_celda + 15))
    
    actual = inicio
    for idx, estanteria in enumerate(orden):
        meta = encontrar_meta_adyacente(almacen, estanteria)
        camino, _ = busqueda_a_estrella(almacen, actual, meta)
        for nodo in camino:
            pygame.draw.rect(pantalla, colores_camino[idx % len(colores_camino)], (nodo[1] * tam_celda, nodo[0] * tam_celda, tam_celda, tam_celda))
        actual = meta
    
    pygame.display.flip()
    print(f"Costo total del recorrido: {costo_total}")
    pygame.time.delay(10000)
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
estanterias = {
        1: (1, 2), 2: (1, 3), 3: (2, 2), 4: (2, 3), 5: (3, 2), 6: (3, 3), 7: (4, 2), 8: (4, 3),
        9: (1, 6), 10: (1, 7), 11: (2, 6), 12: (2, 7), 13: (3, 6), 14: (3, 7), 15: (4, 6), 16: (4, 7),
        17: (1, 10), 18: (1, 11), 19: (2, 10), 20: (2, 11), 21: (3, 10), 22: (3, 11), 23: (4, 10), 24: (4, 11),
        25: (6, 2), 26: (6, 3), 27: (7, 2), 28: (7, 3), 29: (8, 2), 30: (8, 3), 31: (9, 2), 32: (9, 3),
        33: (6, 6), 34: (6, 7), 35: (7, 6), 36: (7, 7), 37: (8, 6), 38: (8, 7), 39: (9, 6), 40: (9, 7),
        41: (6, 10), 42: (6, 11), 43: (7, 10), 44: (7, 11), 45: (8, 10), 46: (8, 11), 47: (9, 10), 48: (9, 11),
    }

posicion_inicial = (5, 0)
num_estanterias = [41,30,11,14,31]
estanterias_a_visitar = [estanterias[n] for n in num_estanterias]
mejor_ruta, costo_total, iteraciones_simulado = temple_simulado(almacen, posicion_inicial, estanterias_a_visitar)
mejor_ruta_numeros = []
for coord in mejor_ruta:
    for num, estanteria_coord in estanterias.items():
        if coord == estanteria_coord:  # Comprobamos si la coordenada está en la mejor ruta
            mejor_ruta_numeros.append(num)
print(f"Iteraciones del Temple Simulado: {iteraciones_simulado}")
dibujar_ruta(almacen, posicion_inicial, mejor_ruta, costo_total, estanterias)
