import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pygame

# =============================================
# CONFIGURACIÓN
# =============================================
g = 9.81
M = 1.0
m = 0.1
l = 0.5
dt = 0.02

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
# VARIABLES DIFUSAS
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
# REGLAS DE CONTROL (AJUSTADAS PARA RANGO AMPLIADO)
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
# FUNCIONES AUXILIARES
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

def modelo_pendulo(theta, theta_punto, F, x, x_punto):
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
            return theta, theta_punto, x, x_punto
            
        theta_segundo = numerador / denominador
        theta_punto_nuevo = theta_punto + theta_segundo * dt
        theta_nuevo = theta + theta_punto * dt + 0.5 * theta_segundo * dt**2
        
        # Movimiento del carro
        x_punto_segundo = (F + m*l*(theta_punto**2*sin_theta - theta_segundo*cos_theta))/(M + m)
        x_punto_nuevo = x_punto + x_punto_segundo * dt
        x_nuevo = x + x_punto * dt + 0.5 * x_punto_segundo * dt**2

        # Normalizar el ángulo
        theta_nuevo = normalizar_angulo(theta_nuevo)

        return theta_nuevo, theta_punto_nuevo, x_nuevo, x_punto_nuevo
    except:
        return normalizar_angulo(theta), theta_punto, x, x_punto

# =============================================
# SIMULACIÓN
# =============================================
def simular(angulo_inicial=0.1, tiempo_total=10):
    # Normalizar ángulo inicial
    theta_actual = normalizar_angulo(angulo_inicial)
    theta_punto_actual = 0.0
    x_actual = 600  # Posición inicial del carro
    x_punto_actual = 0  # Velocidad inicial del carro
    tiempo = np.arange(0, tiempo_total, dt)
    
    historico_theta = []
    historico_fuerza = []
    historico_posicion_carro = []

    for t in tiempo:
        F = control_difuso(theta_actual, theta_punto_actual)
        theta_actual, theta_punto_actual, x_actual, x_punto_actual = modelo_pendulo(theta_actual, theta_punto_actual, F, x_actual, x_punto_actual)
        
        # Guardar el ángulo y la posición del carro
        historico_theta.append(theta_actual)
        historico_fuerza.append(F)
        historico_posicion_carro.append(x_actual)

        # Verificar si el péndulo ha dado una vuelta completa
        if abs(theta_actual) > np.pi - 0.1:  # Pequeño margen para la detección
            print(f"¡Péndulo cambio de signo en t = {t:.2f}s!")

    return tiempo, historico_theta, historico_fuerza, historico_posicion_carro

def animate_pendulum(tiempo, historico_theta, historico_posicion_carro):
    """Crear animación del péndulo invertido con Pygame"""
    pygame.init()
    
    screen_width = 900
    screen_height = 700
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Simulación Péndulo Invertido')

    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)  # Fuente para el texto

    max_frames = len(historico_theta)
    
    i = 0
    while i < max_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Limpiar la pantalla
        screen.fill((255, 255, 255))
        
        if i < len(historico_theta):
            theta = historico_theta[i]
            x = historico_posicion_carro[i]
        else:
            print(f"Error: El índice {i} está fuera del rango de historico_theta.")
            break

        pendulum_length = 150  # longitud del péndulo en píxeles
        pendulum_x = x + pendulum_length * np.sin(theta)
        pendulum_y = 300 - pendulum_length * np.cos(theta)  # La altura se ajusta para que esté dentro de la pantalla
        
        # Dibujar el carrito
        pygame.draw.rect(screen, (0, 0, 255), (x - 50, 290, 100, 20))  # Carro
        
        # Dibujar el péndulo
        pygame.draw.line(screen, (0, 0, 0), (x, 300), (pendulum_x, pendulum_y), 5)  # Cuerda gruesa
        pygame.draw.circle(screen, (255, 0, 0), (int(pendulum_x), int(pendulum_y)), 20)  # Péndulo grande
        
        # Mostrar información (Fuerza, Ángulo, Tiempo)
        angle_text = font.render(f'Ángulo: {np.degrees(theta):.2f}°', True, (0, 0, 0))
        force_text = font.render(f'Fuerza: {historico_fuerza[i]:.2f} N', True, (0, 0, 0))
        time_text = font.render(f'Tiempo: {tiempo[i]:.2f} s', True, (0, 0, 0))

        # Mostrar el texto en la pantalla
        screen.blit(angle_text, (10, 10))
        screen.blit(force_text, (10, 50))
        screen.blit(time_text, (10, 90))

        # Actualizar la pantalla
        pygame.display.flip()

        # Controlar la velocidad de la animación
        clock.tick(60)  # 60 FPS
        i += 1

    pygame.quit()

# ==============================================
# EJECUCIÓN DE SIMULACIÓN Y ANIMACIÓN
# ==============================================
tiempo, historico_theta, historico_fuerza, historico_posicion_carro = simular(angulo_inicial=-np.pi/2, tiempo_total=10)
animate_pendulum(tiempo, historico_theta, historico_posicion_carro)
