import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pygame
import sys

# =============================================
# CONFIGURACIÓN
# =============================================
# Parámetros físicos
g = 9.81
M = 1.0      # Masa del carro
m = 0.1      # Masa del péndulo
l = 0.5      # Longitud del péndulo
dt = 0.02    # Paso de tiempo

# Configuración de Pygame
pygame.init()
ANCHO, ALTO = 1000, 600
PANTALLA = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Péndulo Invertido con Control Difuso")
RELACION = 100  # Píxeles por metro (escala para visualización)

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
ROJO = (255, 0, 0)
AZUL = (0, 0, 255)
VERDE = (0, 128, 0)
GRIS = (200, 200, 200)

# Fuente para texto
FUENTE = pygame.font.SysFont('Arial', 16)

# Rangos ampliados a ±180° (π radianes)
theta_range = np.arange(-np.pi, np.pi, 0.01)  # -180° a 180°
theta_punto_range = np.arange(-10, 10, 0.1)   # Rango de velocidad angular
fuerza_range = np.arange(-30, 30, 0.5)        # Rango de fuerza

# =============================================
# FUNCIÓN DE NORMALIZACIÓN DE ÁNGULOS
# =============================================
def normalizar_angulo(angulo):
    """Normaliza el ángulo al rango [-π, π]"""
    return (angulo + np.pi) % (2 * np.pi) - np.pi

# =============================================
# VARIABLES DIFUSAS (igual que en tu código original)
# =============================================
theta = ctrl.Antecedent(theta_range, 'theta')
theta_punto = ctrl.Antecedent(theta_punto_range, 'theta_punto')
fuerza = ctrl.Consequent(fuerza_range, 'fuerza')

# Funciones de membresía para theta (±180°)
theta['NG'] = fuzz.trapmf(theta_range, [-np.pi, -np.pi, -np.pi/2, -np.pi/4])
theta['NP'] = fuzz.trimf(theta_range, [-np.pi/2, -np.pi/6, 0])
theta['CE'] = fuzz.trimf(theta_range, [-np.pi/12, 0, np.pi/12])
theta['PP'] = fuzz.trimf(theta_range, [0, np.pi/6, np.pi/2])
theta['PG'] = fuzz.trapmf(theta_range, [np.pi/4, np.pi/2, np.pi, np.pi])

# Funciones de membresía para theta_punto (sin cambios)
theta_punto['NG'] = fuzz.trimf(theta_punto_range, [-10, -10, -5])
theta_punto['NP'] = fuzz.trimf(theta_punto_range, [-7, -3, 0])
theta_punto['CE'] = fuzz.trimf(theta_punto_range, [-2, 0, 2])
theta_punto['PP'] = fuzz.trimf(theta_punto_range, [0, 3, 7])
theta_punto['PG'] = fuzz.trimf(theta_punto_range, [5, 10, 10])

# Funciones de membresía para fuerza (sin cambios)
fuerza['NG'] = fuzz.trimf(fuerza_range, [-30, -30, -15])
fuerza['NP'] = fuzz.trimf(fuerza_range, [-20, -10, 0])
fuerza['CE'] = fuzz.trimf(fuerza_range, [-5, 0, 5])
fuerza['PP'] = fuzz.trimf(fuerza_range, [0, 10, 20])
fuerza['PG'] = fuzz.trimf(fuerza_range, [15, 30, 30])

# =============================================
# REGLAS DE CONTROL (igual que en tu código original)
# =============================================
reglas = []
reglas_matrix = [
    ['NG', 'NG', 'NG'],
    ['NG', 'NP', 'NG'],
    ['NG', 'CE', 'NG'],
    ['NG', 'PP', 'NP'],
    ['NG', 'PG', 'CE'],

    ['NP', 'NG', 'NG'],
    ['NP', 'NP', 'NG'],
    ['NP', 'CE', 'NP'],
    ['NP', 'PP', 'CE'],
    ['NP', 'PG', 'PP'],

    ['CE', 'NG', 'NG'],
    ['CE', 'NP', 'NP'],
    ['CE', 'CE', 'CE'],
    ['CE', 'PP', 'PP'],
    ['CE', 'PG', 'PG'],

    ['PP', 'NG', 'NP'],
    ['PP', 'NP', 'CE'],
    ['PP', 'CE', 'PP'],
    ['PP', 'PP', 'PG'],
    ['PP', 'PG', 'PG'],

    ['PG', 'NG', 'CE'],
    ['PG', 'NP', 'PP'],
    ['PG', 'CE', 'PG'],
    ['PG', 'PP', 'PG'],
    ['PG', 'PG', 'PG']
]

for rule in reglas_matrix:
    reglas.append(ctrl.Rule(
        theta[rule[0]] & theta_punto[rule[1]], 
        fuerza[rule[2]],
        label=f"Regla_{rule[0]}_{rule[1]}"
    ))

# =============================================
# SISTEMA DE CONTROL
# =============================================
sistema_control = ctrl.ControlSystem(reglas)
simulador = ctrl.ControlSystemSimulation(sistema_control)

# =============================================
# FUNCIONES AUXILIARES (con pequeñas mejoras)
# =============================================
def control_difuso(theta_val, theta_punto_val):
    """Función segura para calcular la fuerza con normalización de ángulo"""
    try:
        # Normalizar el ángulo primero
        theta_val = normalizar_angulo(theta_val)
        
        # Limitar valores dentro de los rangos
        theta_val = np.clip(theta_val, -np.pi, np.pi)
        theta_punto_val = np.clip(theta_punto_val, -10, 10)
        
        simulador.input['theta'] = theta_val
        simulador.input['theta_punto'] = theta_punto_val
        simulador.compute()
        return simulador.output['fuerza']
    except:
        # Valor por defecto si hay error
        return 0.0

def modelo_pendulo(theta, theta_punto, F):
    """Modelo físico con protección contra valores extremos y normalización de ángulo"""
    try:
        # Normalizar el ángulo primero
        theta = normalizar_angulo(theta)
        
        # Calcular componentes reutilizables
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        numerador = g * sin_theta + cos_theta * ((-F - m*l*theta_punto**2*sin_theta)/(M + m))
        denominador = l * (4/3 - (m*cos_theta**2)/(M + m))
        
        # Protección contra división por cero
        if abs(denominador) < 1e-6:
            return theta, theta_punto
            
        theta_segundo = numerador / denominador
        theta_punto_nuevo = theta_punto + theta_segundo * dt
        theta_nuevo = theta + theta_punto * dt + 0.5 * theta_segundo * dt**2
        
        # Normalizar el nuevo ángulo
        theta_nuevo = normalizar_angulo(theta_nuevo)
        
        return theta_nuevo, theta_punto_nuevo
    except:
        return normalizar_angulo(theta), theta_punto

# =============================================
# FUNCIÓN PARA DIBUJAR EL PÉNDULO EN PYGAME
# =============================================
def dibujar_pendulo(pantalla, x_carro, theta, fuerza, tiempo, ancho_carro=50, alto_carro=30):
    """Dibuja el carro y el péndulo en la pantalla de Pygame"""
    pantalla.fill(BLANCO)
    
    # Convertir coordenadas físicas a píxeles
    x_centro = ANCHO // 2
    y_piso = ALTO - 100
    
    # Dibujar piso
    pygame.draw.line(pantalla, NEGRO, (0, y_piso), (ANCHO, y_piso), 2)
    
    # Calcular posición del carro (limitada a los bordes de la pantalla)
    x_carro_pix = int(x_centro + x_carro * RELACION)
    x_carro_pix = max(ancho_carro//2, min(ANCHO - ancho_carro//2, x_carro_pix))
    
    # Dibujar carro
    pygame.draw.rect(pantalla, AZUL, 
                    (x_carro_pix - ancho_carro//2, y_piso - alto_carro, 
                     ancho_carro, alto_carro), 0, 5)
    
    # Calcular posición del péndulo
    x_pend = x_carro_pix + np.sin(theta) * l * RELACION
    y_pend = y_piso - alto_carro - np.cos(theta) * l * RELACION
    
    # Dibujar péndulo
    pygame.draw.line(pantalla, NEGRO, 
                    (x_carro_pix, y_piso - alto_carro), 
                    (x_pend, y_pend), 3)
    pygame.draw.circle(pantalla, ROJO, (int(x_pend), int(y_pend)), 10)
    
    # Dibujar información
    texto_angulo = FUENTE.render(f"Ángulo: {np.degrees(theta):.1f}°", True, NEGRO)
    texto_fuerza = FUENTE.render(f"Fuerza: {fuerza:.2f} N", True, NEGRO)
    texto_tiempo = FUENTE.render(f"Tiempo: {tiempo:.2f} s", True, NEGRO)
    
    pantalla.blit(texto_angulo, (20, 20))
    pantalla.blit(texto_fuerza, (20, 50))
    pantalla.blit(texto_tiempo, (20, 80))
    
    # Indicador de fuerza
    if fuerza != 0:
        color_fuerza = VERDE if fuerza > 0 else ROJO
        direccion = 1 if fuerza > 0 else -1
        longitud = min(100, abs(fuerza) * 5)
        
        pygame.draw.line(pantalla, color_fuerza,
                        (x_carro_pix, y_piso + 20),
                        (x_carro_pix + direccion * longitud, y_piso + 20), 5)
        pygame.draw.polygon(pantalla, color_fuerza, [
            (x_carro_pix + direccion * longitud, y_piso + 15),
            (x_carro_pix + direccion * longitud, y_piso + 25),
            (x_carro_pix + direccion * (longitud + 10), y_piso + 20)
        ])
    
    pygame.display.flip()

# =============================================
# SIMULACIÓN CON ANIMACIÓN EN PYGAME
# =============================================
def simular_con_animacion(angulo_inicial=0.1, tiempo_total=10):
    """Ejecuta la simulación con animación en Pygame y gráficas en matplotlib"""
    # Inicialización
    theta_actual = normalizar_angulo(angulo_inicial)
    theta_punto_actual = 0.0
    x_carro = 0.0  # Posición horizontal del carro
    x_punto_carro = 0.0  # Velocidad del carro
    
    tiempo = np.arange(0, tiempo_total, dt)
    
    # Datos para gráficos
    historico_theta = []
    historico_fuerza = []
    historico_x_carro = []
    historico_tiempo = []
    
    # Mostrar funciones de membresía en ventana aparte
    plt.figure(figsize=(10, 8))
    theta.view()
    plt.title("Funciones de membresía para el ángulo")
    plt.show(block=False)
    
    # Bucle principal de simulación
    reloj = pygame.time.Clock()
    ejecutando = True
    pausa = False
    
    for t in tiempo:
        # Manejo de eventos de Pygame
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE:
                    pausa = not pausa
                elif evento.key == pygame.K_ESCAPE:
                    ejecutando = False
        
        if not ejecutando:
            break
        
        if pausa:
            reloj.tick(60)
            continue
        
        # Calcular fuerza de control
        F = control_difuso(theta_actual, theta_punto_actual)
        
        # Actualizar modelo físico
        theta_actual, theta_punto_actual = modelo_pendulo(theta_actual, theta_punto_actual, F)
        
        # Actualizar posición del carro (modelo simplificado)
        x_punto_punto_carro = F / (M + m)  # F = ma
        x_punto_carro += x_punto_punto_carro * dt
        x_carro += x_punto_carro * dt
        
        # Guardar datos
        historico_theta.append(theta_actual)
        historico_fuerza.append(F)
        historico_x_carro.append(x_carro)
        historico_tiempo.append(t)
        
        # Dibujar en Pygame
        dibujar_pendulo(PANTALLA, x_carro, theta_actual, F, t)
        
        # Controlar velocidad de la simulación
        reloj.tick(1/dt)  # Intentar mantener la velocidad real
        
    # Generar gráficos después de la simulación
    generar_graficos(historico_tiempo, historico_theta, historico_fuerza, historico_x_carro, angulo_inicial)
    
    pygame.quit()

def generar_graficos(tiempo, theta, fuerza, x_carro, angulo_inicial):
    """Genera las gráficas de resultados en ventanas separadas"""
    # Gráfico 1: Ángulo y posición del carro
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(tiempo, np.degrees(theta), 'b', label='Ángulo (grados)')
    plt.ylabel('Ángulo (grados)')
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(tiempo, x_carro, 'g', label='Posición del carro (m)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición (m)')
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(f'Comportamiento del Péndulo Invertido (θ₀ = {np.degrees(angulo_inicial):.1f}°)')
    
    # Gráfico 2: Fuerza aplicada
    plt.figure(figsize=(12, 4))
    plt.plot(tiempo, fuerza, 'r', label='Fuerza aplicada (N)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Fuerza (N)')
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True)
    plt.title('Fuerza de Control Aplicada')
    
    plt.show()

# =============================================
# EJECUCIÓN PRINCIPAL
# =============================================
if __name__ == "__main__":
    print("Iniciando simulación con animación Pygame...")
    
    # Ejecutar simulaciones con diferentes ángulos iniciales
    try:
        # Simulación 1: Pequeña perturbación
        print("\nSimulación 1: Ángulo inicial = 180°")
        simular_con_animacion(angulo_inicial=np.radians(180), tiempo_total=5)
        
        
    except Exception as e:
        print(f"Error durante la simulación: {str(e)}")
    finally:
        pygame.quit()
        sys.exit()