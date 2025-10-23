import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import j0  # Para comparar con la solución analítica J0

# ======================================
# 1. Configuración básica
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {device}")

# Orden de la función de Bessel (n=0)
n = 0

# Intervalo de entrenamiento para rho
rho_min = 0.0
rho_max = 10.0

# Número de puntos de entrenamiento (colocación)
N_col = 2000

# Hiperparámetros de la red y entrenamiento
num_epochs = 7500
lr = 0.004


# ======================================
# 2. Definición de la red neuronal
# ======================================

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class BesselNN(nn.Module):
    def __init__(self, hidden_layers=3, hidden_neurons=20):
        super(BesselNN, self).__init__()

        layers = []
        input_size = 1
        output_size = 1

        # Capa de entrada
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(Sin())

        # Capas ocultas
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(Sin())

        # Capa de salida
        layers.append(nn.Linear(hidden_neurons, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Instanciamos la red
model = BesselNN(hidden_layers=3, hidden_neurons=20).to(device)

# ======================================
# 3. Funciones auxiliares para derivadas
# ======================================
def first_derivative(model, x):
    # NO usar clone().detach()
    # Basta con asumir que x.requires_grad_(True) se hace afuera
    R = model(x)
    dRdx = torch.autograd.grad(
        R, x,
        grad_outputs=torch.ones_like(R),
        create_graph=True
    )[0]
    return dRdx

def second_derivative(model, x):
    # Calculamos R y su primera derivada sin romper la conexión
    R = model(x)
    dRdx = torch.autograd.grad(
        R, x,
        grad_outputs=torch.ones_like(R),
        create_graph=True
    )[0]

    # dRdx sigue conectado al grafo
    d2Rdx2 = torch.autograd.grad(
        dRdx, x,
        grad_outputs=torch.ones_like(dRdx),
        create_graph=True
    )[0]
    return d2Rdx2

# ======================================
# 4. Definición de la función de pérdida (residuo + frontera)
# ======================================

def loss_function(model, rho_col):
    rho_col.requires_grad_(True)

    R_pred = model(rho_col)
    dR = first_derivative(model, rho_col)
    d2R = second_derivative(model, rho_col)

    # Ecuación de Bessel (n=0) -> residual
    residual = (rho_col**2)*d2R + rho_col*dR + (rho_col**2)*R_pred
    loss_pde = torch.mean(residual**2)

    # Condiciones de frontera en rho=0
    eps = torch.tensor([1e-5], dtype=torch.float32, device=device, requires_grad=True)
    R_bc0 = model(eps)
    dR_bc0 = first_derivative(model, eps)

    loss_bc = (R_bc0 - 1.0)**2 + (dR_bc0 - 0.0)**2

    return loss_pde + torch.mean(loss_bc)

# ======================================
# 5. Generación de datos (SOLO puntos de colación)
#    — sin cambiar nada más del código —
# ======================================

# Evita rho=0 exacto por la singularidad numérica
eps_np = 1e-5

# Elige la estrategia de muestreo:
#   "uniform"  -> np.linspace clásico (tu versión original)
#   "power"    -> concentra puntos cerca de 0 con t**alpha  (recomendado)
#   "chebyshev"-> más denso en extremos (0 y rho_max)
#   "sobol"    -> muestreo cuasi-aleatorio (requiere scipy>=1.7)
COLLOCATION_STRATEGY = "power"   # <- cambia aquí si quieres probar otra

# Parámetros para estrategias
alpha = 4.0   # potencia para "power" (2–4 suele ir bien)

if COLLOCATION_STRATEGY == "uniform":
    rho_col_np = np.linspace(eps_np, rho_max, N_col)

elif COLLOCATION_STRATEGY == "power":
    # t in [0,1] -> mapea con t**alpha para concentrar cerca de 0
    t = np.linspace(0.0, 1.0, N_col)
    rho_col_np = eps_np + (rho_max - eps_np) * (t ** alpha)

elif COLLOCATION_STRATEGY == "chebyshev":
    # Chebyshev mapeado a [eps, rho_max] (más puntos en extremos)
    i = np.arange(1, N_col + 1)
    x = np.cos((2 * i - 1) * np.pi / (2 * N_col))  # en [-1,1]
    rho_col_np = eps_np + (rho_max - eps_np) * (x + 1.0) / 2.0
    rho_col_np = np.sort(rho_col_np)  # orden ascendente para estabilidad

elif COLLOCATION_STRATEGY == "sobol":
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=1, scramble=True)
        rho_col_np = eps_np + (rho_max - eps_np) * sampler.random(N_col).flatten()
        rho_col_np = np.sort(rho_col_np)
    except Exception as e:
        print(f"[AVISO] Sobol no disponible ({e}). Usando uniforme.")
        rho_col_np = np.linspace(eps_np, rho_max, N_col)

else:
    raise ValueError("COLLOCATION_STRATEGY no reconocida.")

# Tensor final de colación
rho_col = torch.tensor(rho_col_np, dtype=torch.float32).view(-1, 1).to(device)


optimizer = optim.Adam(model.parameters(), lr=lr)

loss_history = []

print("Iniciando entrenamiento...")
# Medir tiempo de entrenamiento de la PINN
start_time_pinn = time.time()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_function(model, rho_col)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Pérdida: {loss.item():.9f}")

end_time_pinn = time.time()
training_time_pinn = end_time_pinn - start_time_pinn
print(f"Entrenamiento completado en {training_time_pinn:.4f} segundos\n")

# ======================================
# 6. Evaluación y comparación con solución analítica
# ======================================
model.eval()

# Puntos de prueba para graficar
rho_test_np = np.linspace(0, rho_max, 300)
rho_test = torch.tensor(rho_test_np, dtype=torch.float32).view(-1,1).to(device)

# Medir tiempo de inferencia de la PINN
start_inference = time.time()
with torch.no_grad():
    R_pred_test = model(rho_test).cpu().numpy().flatten()
end_inference = time.time()
inference_time_pinn = end_inference - start_inference

# Solución analítica (Bessel J0)
R_exact = j0(rho_test_np)

# ======================================
# MÉTODO NUMÉRICO DE EULER
# ======================================
print("Ejecutando método de Euler...")
start_time_euler = time.time()

def euler_bessel_j0(rho_array, h=None):
    """
    Resuelve la ecuación de Bessel J0 usando el método de Euler.
    Ecuación: rho^2 R'' + rho R' + rho^2 R = 0
    Sistema de primer orden:
        R' = S
        S' = -(rho S + rho^2 R) / rho^2 = -S/rho - R

    Condiciones iniciales:
        R(eps) = 1.0
        S(eps) = 0.0 (derivada en rho≈0)
    """
    n_points = len(rho_array)

    # Paso de integración (si no se especifica, usar el espaciado del array)
    if h is None:
        h = rho_array[1] - rho_array[0]

    # Inicializar arrays
    R = np.zeros(n_points)
    S = np.zeros(n_points)  # S = R'

    # Condiciones iniciales (en rho muy pequeño, cerca de 0)
    R[0] = 1.0
    S[0] = 0.0

    # Método de Euler
    for i in range(n_points - 1):
        rho_i = rho_array[i]

        # Evitar división por cero
        if rho_i < 1e-10:
            rho_i = 1e-10

        # Derivadas
        dR = S[i]
        dS = -S[i]/rho_i - R[i]

        # Actualización de Euler
        R[i+1] = R[i] + h * dR
        S[i+1] = S[i] + h * dS

    return R

# Ejecutar Euler con los mismos puntos de prueba
R_euler = euler_bessel_j0(rho_test_np)

end_time_euler = time.time()
computation_time_euler = end_time_euler - start_time_euler

print(f"Método de Euler completado en {computation_time_euler:.6f} segundos\n")

# ======================================
# MÉTRICAS CUANTITATIVAS AGREGADAS
# ======================================

# 1. Errores absolutos
mse = np.mean((R_pred_test - R_exact)**2)
rmse = np.sqrt(mse)
max_error = np.max(np.abs(R_pred_test - R_exact))
mean_abs_error = np.mean(np.abs(R_pred_test - R_exact))

# 2. Error relativo promedio y similitud (PROBLEMA: sensible a cruces por cero)
denominator = np.maximum(np.abs(R_exact), 1e-10)
relative_error = np.abs(R_pred_test - R_exact) / denominator
mean_relative_error = np.mean(relative_error)
similarity = (1 - mean_relative_error) * 100

# 3. MAPE (Mean Absolute Percentage Error)
mape = np.mean(relative_error) * 100

# 3b. Error relativo ROBUSTO (ignora puntos donde |exact| < umbral)
threshold = 0.1  # Ignorar donde |exact| < 0.1
mask_robust = np.abs(R_exact) > threshold
if np.any(mask_robust):
    relative_error_robust = np.abs(R_pred_test[mask_robust] - R_exact[mask_robust]) / np.abs(R_exact[mask_robust])
    mean_relative_error_robust = np.mean(relative_error_robust)
    similarity_robust = (1 - mean_relative_error_robust) * 100
else:
    similarity_robust = 0.0

# 4. R² (Coeficiente de determinación)
ss_res = np.sum((R_exact - R_pred_test)**2)
ss_tot = np.sum((R_exact - np.mean(R_exact))**2)
r2_score = 1 - (ss_res / ss_tot)

# 5. Correlación de Pearson
correlation = np.corrcoef(R_exact, R_pred_test)[0, 1]

# 6. Error normalizado por el rango
range_exact = np.max(R_exact) - np.min(R_exact)
normalized_rmse = rmse / range_exact if range_exact > 0 else 0
nrmse_percent = normalized_rmse * 100

# ======================================
# MÉTRICAS DEL MÉTODO DE EULER
# ======================================
mse_euler = np.mean((R_euler - R_exact)**2)
rmse_euler = np.sqrt(mse_euler)
max_error_euler = np.max(np.abs(R_euler - R_exact))
mean_abs_error_euler = np.mean(np.abs(R_euler - R_exact))

denominator_euler = np.maximum(np.abs(R_exact), 1e-10)
relative_error_euler = np.abs(R_euler - R_exact) / denominator_euler
mean_relative_error_euler = np.mean(relative_error_euler)
similarity_euler = (1 - mean_relative_error_euler) * 100
mape_euler = mean_relative_error_euler * 100

# Error relativo ROBUSTO para Euler
if np.any(mask_robust):
    relative_error_euler_robust = np.abs(R_euler[mask_robust] - R_exact[mask_robust]) / np.abs(R_exact[mask_robust])
    mean_relative_error_euler_robust = np.mean(relative_error_euler_robust)
    similarity_euler_robust = (1 - mean_relative_error_euler_robust) * 100
else:
    similarity_euler_robust = 0.0

ss_res_euler = np.sum((R_exact - R_euler)**2)
ss_tot_euler = np.sum((R_exact - np.mean(R_exact))**2)
r2_score_euler = 1 - (ss_res_euler / ss_tot_euler)

correlation_euler = np.corrcoef(R_exact, R_euler)[0, 1]
nrmse_euler = (rmse_euler / range_exact * 100) if range_exact > 0 else 0

# Imprimir resultados comparativos
print("="*70)
print(" "*20 + "COMPARACIÓN DE MÉTODOS")
print("="*70)
print()

print("="*70)
print(" "*22 + "PINN (Red Neuronal)")
print("="*70)
print(f"MSE (Error Cuadrático Medio):              {mse:.8f}")
print(f"RMSE (Raíz del MSE):                       {rmse:.8f}")
print(f"Error Máximo:                              {max_error:.8f}")
print(f"Error Absoluto Promedio:                   {mean_abs_error:.8f}")
print("-"*70)
print(f"MAPE (Mean Absolute Percentage Error):     {mape:.6f}%")
print(f"Error Relativo Promedio:                   {mean_relative_error:.8f}")
print(f"SIMILITUD (1 - Error Relativo):            {similarity:.6f}%")
print(f"SIMILITUD ROBUSTA (|exact|>0.1):           {similarity_robust:.6f}%  ⭐")
print("-"*70)
print(f"R² (Coeficiente de Determinación):         {r2_score:.8f}")
print(f"Correlación de Pearson:                    {correlation:.8f}")
print(f"NRMSE (Error Normalizado por Rango):       {nrmse_percent:.6f}%")
print("="*70)
print()

print("="*70)
print(" "*22 + "MÉTODO DE EULER")
print("="*70)
print(f"MSE (Error Cuadrático Medio):              {mse_euler:.8f}")
print(f"RMSE (Raíz del MSE):                       {rmse_euler:.8f}")
print(f"Error Máximo:                              {max_error_euler:.8f}")
print(f"Error Absoluto Promedio:                   {mean_abs_error_euler:.8f}")
print("-"*70)
print(f"MAPE (Mean Absolute Percentage Error):     {mape_euler:.6f}%")
print(f"Error Relativo Promedio:                   {mean_relative_error_euler:.8f}")
print(f"SIMILITUD (1 - Error Relativo):            {similarity_euler:.6f}%")
print(f"SIMILITUD ROBUSTA (|exact|>0.1):           {similarity_euler_robust:.6f}%  ⭐")
print("-"*70)
print(f"R² (Coeficiente de Determinación):         {r2_score_euler:.8f}")
print(f"Correlación de Pearson:                    {correlation_euler:.8f}")
print(f"NRMSE (Error Normalizado por Rango):       {nrmse_euler:.6f}%")
print("="*70)
print()

print("="*70)
print(" "*15 + "COMPARACIÓN DE COMPLEJIDAD COMPUTACIONAL")
print("="*70)
print(f"Tiempo de entrenamiento PINN:              {training_time_pinn:.6f} segundos")
print(f"Tiempo de inferencia PINN:                 {inference_time_pinn:.6f} segundos")
print(f"Tiempo TOTAL PINN:                         {training_time_pinn + inference_time_pinn:.6f} segundos")
print("-"*70)
print(f"Tiempo de cómputo Euler:                   {computation_time_euler:.6f} segundos")
print("-"*70)
print(f"Speedup de inferencia (Euler/PINN):        {computation_time_euler/inference_time_pinn:.2f}x")
print()
print("NOTA: La PINN requiere entrenamiento inicial (una sola vez),")
print("      pero luego la inferencia es muy rápida para nuevos puntos.")
print("      El método de Euler debe recalcular todo desde cero cada vez.")
print("="*70)
print()
print("⚠️  IMPORTANTE: La 'Similitud Robusta' ignora puntos donde |exact|<0.1")
print("    para evitar que los cruces por cero distorsionen la métrica.")
print("    Esta métrica refleja mejor la precisión visual de las curvas.")
print()

# Gráfica de comparación entre los 3 métodos
plt.figure(figsize=(14,6))

# Subplot 1: Comparación de soluciones
plt.subplot(1, 2, 1)
plt.plot(rho_test_np, R_exact, label="J0 Analítico", linewidth=2.5, color='black')
plt.plot(rho_test_np, R_pred_test, '--', label=f"PINN (Sim Robusta: {similarity_robust:.2f}%)", linewidth=2, color='blue')
plt.plot(rho_test_np, R_euler, ':', label=f"Euler (Sim Robusta: {similarity_euler_robust:.2f}%)", linewidth=2, color='red')
plt.xlabel("rho", fontsize=12)
plt.ylabel("R(rho)", fontsize=12)
plt.title("Comparación: PINN vs Euler vs Analítico", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Subplot 2: Comparación de errores absolutos
plt.subplot(1, 2, 2)
plt.plot(rho_test_np, np.abs(R_pred_test - R_exact), label="Error PINN", linewidth=2, color='blue')
plt.plot(rho_test_np, np.abs(R_euler - R_exact), label="Error Euler", linewidth=2, color='red')
plt.xlabel("rho", fontsize=12)
plt.ylabel("|Error|", fontsize=12)
plt.title("Comparación de Errores Absolutos", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()

# Gráfica de comparación de tiempos computacionales
plt.figure(figsize=(10,6))
methods = ['PINN\n(Inferencia)', 'Euler']
times = [inference_time_pinn, computation_time_euler]
colors = ['blue', 'red']

bars = plt.bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Agregar valores en las barras
for i, (bar, time_val) in enumerate(zip(bars, times)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{time_val:.6f}s',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel("Tiempo (segundos)", fontsize=12)
plt.title("Complejidad Computacional: PINN vs Euler\n(Inferencia en 300 puntos)", fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Evolución de la pérdida
plt.figure(figsize=(10,6))
plt.plot(loss_history, linewidth=2)
plt.xlabel("Época", fontsize=12)
plt.ylabel("Pérdida", fontsize=12)
plt.title("Evolución de la pérdida durante el entrenamiento", fontsize=14)
plt.grid(True)
plt.yscale('log')
plt.tight_layout()
plt.show()

