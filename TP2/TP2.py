import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# =============================================
# 1. PARÁMETROS FÍSICOS
# =============================================
gravedad = 9.81       # Aceleración gravitacional [m/s²]
masa_carro = 1.0      # Masa del carro [kg]
masa_pendulo = 0.2    # Masa del péndulo [kg]
longitud = 0.5        # Longitud del péndulo [m]
paso_tiempo = 0.01    # Paso de tiempo [s]
tiempo_simulacion = 15 # Tiempo total de simulación [s]
pasos = int(tiempo_simulacion / paso_tiempo)

# =============================================
# 2. DEFINICIÓN DE VARIABLES DIFUSAS
# =============================================
angulo = ctrl.Antecedent(np.arange(-2.5*np.pi, 2.5*np.pi, 0.01), 'ángulo')
velocidad_angular = ctrl.Antecedent(np.arange(-10, 10, 0.1), 'velocidad_angular')
fuerza = ctrl.Consequent(np.arange(-50, 50, 0.5), 'fuerza')


# Funciones de membresía para el ángulo
angulo['NG'] = fuzz.trapmf(angulo.universe, [-2.5*np.pi, -2.5*np.pi, -np.pi, -np.radians(150)])
angulo['NP'] = fuzz.trimf(angulo.universe, [-np.pi, -np.radians(90), 0])
angulo['Z'] = fuzz.trimf(angulo.universe, [-np.radians(30), 0, np.radians(30)])
angulo['PP'] = fuzz.trimf(angulo.universe, [0, np.radians(90), np.pi])
angulo['PG'] = fuzz.trapmf(angulo.universe, [np.radians(150), np.pi, 2.5*np.pi, 2.5*np.pi])

# Funciones de membresía para velocidad angular
velocidad_angular['NG'] = fuzz.trapmf(velocidad_angular.universe, [-10, -10, -4, -1])
velocidad_angular['NP'] = fuzz.trimf(velocidad_angular.universe, [-4, -1, 0])
velocidad_angular['Z'] = fuzz.trimf(velocidad_angular.universe, [-1, 0, 1])
velocidad_angular['PP'] = fuzz.trimf(velocidad_angular.universe, [0, 1, 4])
velocidad_angular['PG'] = fuzz.trapmf(velocidad_angular.universe, [1, 4, 10, 10])

# Funciones de membresía para fuerza
fuerza['NG'] = fuzz.trapmf(fuerza.universe, [-50, -50, -20, -10])
fuerza['NP'] = fuzz.trimf(fuerza.universe, [-20, -10, 0])
fuerza['Z'] = fuzz.trimf(fuerza.universe, [-10, 0, 10])
fuerza['PP'] = fuzz.trimf(fuerza.universe, [0, 10, 20])
fuerza['PG'] = fuzz.trapmf(fuerza.universe, [10, 20, 50, 50])

# =============================================
# 3. REGLAS DIFUSAS
# =============================================
reglas = [
    ctrl.Rule(angulo['NG'] & velocidad_angular['NG'], fuerza['PG']),
    ctrl.Rule(angulo['NG'] & velocidad_angular['NP'], fuerza['PG']),
    ctrl.Rule(angulo['NG'] & velocidad_angular['Z'], fuerza['PP']),
    ctrl.Rule(angulo['NP'] & velocidad_angular['NG'], fuerza['PG']),
    ctrl.Rule(angulo['NP'] & velocidad_angular['NP'], fuerza['PP']),
    ctrl.Rule(angulo['NP'] & velocidad_angular['Z'], fuerza['PP']),
    ctrl.Rule(angulo['Z'] & velocidad_angular['NG'], fuerza['PP']),
    ctrl.Rule(angulo['Z'] & velocidad_angular['NP'], fuerza['PP']),
    ctrl.Rule(angulo['Z'] & velocidad_angular['Z'], fuerza['Z']),
    ctrl.Rule(angulo['PP'] & velocidad_angular['Z'], fuerza['NP']),
    ctrl.Rule(angulo['PP'] & velocidad_angular['PP'], fuerza['NP']),
    ctrl.Rule(angulo['PP'] & velocidad_angular['PG'], fuerza['NG']),
    ctrl.Rule(angulo['PG'] & velocidad_angular['Z'], fuerza['NP']),
    ctrl.Rule(angulo['PG'] & velocidad_angular['PP'], fuerza['NG']),
    ctrl.Rule(angulo['PG'] & velocidad_angular['PG'], fuerza['NG'])
]

# =============================================
# 4. SISTEMA DE CONTROL
# =============================================
sistema_control = ctrl.ControlSystem(reglas)
simulador = ctrl.ControlSystemSimulation(sistema_control)

# =============================================
# 5. FUNCIONES AUXILIARES
# =============================================
def normalizar_angulo(angulo_rad):
    """Convierte cualquier ángulo al rango [-π, π]"""
    return (angulo_rad + np.pi) % (2*np.pi) - np.pi

def calcular_aceleracion_angular(angulo_rad, velocidad, fuerza_aplicada):
    """Calcula la aceleración angular usando la dinámica del péndulo"""
    angulo_norm = normalizar_angulo(angulo_rad)
    numerador = gravedad * np.sin(angulo_norm) + np.cos(angulo_norm) * ((-fuerza_aplicada - masa_pendulo*longitud*velocidad**2*np.sin(angulo_norm))/(masa_carro + masa_pendulo))
    denominador = longitud * (4/3 - (masa_pendulo * np.cos(angulo_norm)**2)/(masa_carro + masa_pendulo))
    return numerador / denominador

def calcular_fuerza(angulo_rad, velocidad):
    """Calcula la fuerza aplicada usando lógica difusa"""
    angulo_norm = normalizar_angulo(angulo_rad)
    
    # Asignar valores a ambos antecedentes
    simulador.input['ángulo'] = angulo_norm
    simulador.input['velocidad_angular'] = velocidad
    
    try:
        simulador.compute()
        return simulador.output['fuerza']
    except:
        # En caso de error, retornar fuerza cero
        return 0

# =============================================
# 6. SIMULACIÓN
# =============================================
def simular(angulo_inicial_grados=175, velocidad_inicial=0):
    angulo_actual = np.radians(angulo_inicial_grados)
    velocidad_actual = velocidad_inicial
    
    historial_angulos = []
    historial_velocidades = []
    historial_fuerzas = []
    tiempos = np.arange(0, tiempo_simulacion, paso_tiempo)
    
    for _ in range(pasos):
        # Calcular fuerza
        fuerza_aplicada = calcular_fuerza(angulo_actual, velocidad_actual)
        
        # Dinámica del sistema
        aceleracion_angular = calcular_aceleracion_angular(angulo_actual, velocidad_actual, fuerza_aplicada)
        velocidad_actual += aceleracion_angular * paso_tiempo
        angulo_actual += velocidad_actual * paso_tiempo + 0.5 * aceleracion_angular * paso_tiempo**2
        
        # Guardar datos
        historial_angulos.append(np.degrees(normalizar_angulo(angulo_actual)))
        historial_velocidades.append(np.degrees(velocidad_actual))
        historial_fuerzas.append(fuerza_aplicada)
    
    # Visualización
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(tiempos, historial_angulos, 'b', linewidth=2)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Ángulo del Péndulo')
    plt.ylabel('Grados')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(tiempos, historial_velocidades, 'g', linewidth=2)
    plt.title('Velocidad Angular')
    plt.ylabel('Grados/s')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(tiempos, historial_fuerzas, 'r', linewidth=2)
    plt.title('Fuerza Aplicada')
    plt.ylabel('Newtons')
    plt.xlabel('Tiempo (s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================
# 7. EJECUCIÓN
# =============================================
simular(angulo_inicial_grados=175)  # Prueba con ángulo cercano a 180°