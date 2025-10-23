import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import jv

# ======================================
# 1. Configuración básica
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {device}")

N_ORDERS = [0, 1, 2, 3]

rho_min = 0.0
rho_max = 10.0

N_col_per_order = 2500

# MEJORA: Épocas específicas por orden (Curriculum Learning)
EPOCHS_PER_PHASE = {
    'phase_1_J0': 5000,      # Solo J0 (base)
    'phase_2_J0J1': 6000,    # J0 + J1
    'phase_3_J0J1J2': 8000,  # J0 + J1 + J2
    'phase_4_all': 12000,    # Todos juntos
    'phase_5_finetune': 4000 # Fine-tuning final
}

lr_initial = 0.004
alpha_power = 4.5

# Pesos balanceados
LOSS_WEIGHTS = {
    0: 1.0,
    1: 2.5,
    2: 3.5,
    3: 4.5
}


# ======================================
# 2. Red neuronal (arquitectura probada)
# ======================================

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class BesselGeneralNN_v3(nn.Module):
    def __init__(self, hidden_layers=4, hidden_neurons=28):
        super(BesselGeneralNN_v3, self).__init__()

        layers = []
        input_size = 2
        output_size = 1

        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(Sin())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(Sin())

        layers.append(nn.Linear(hidden_neurons, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

model = BesselGeneralNN_v3(hidden_layers=4, hidden_neurons=28).to(device)

print(f"\n🔧 Arquitectura: 4 capas × 28 neuronas")
print(f"   Parámetros: {sum(p.numel() for p in model.parameters())}")


# ======================================
# 3. Funciones auxiliares
# ======================================
def derivative_wrt_rho(model, inputs):
    inputs.requires_grad_(True)
    R = model(inputs)
    dRdx = torch.autograd.grad(R, inputs, grad_outputs=torch.ones_like(R),
                                create_graph=True, retain_graph=True)[0]
    return dRdx[:, 0:1]

def second_derivative_wrt_rho(model, inputs):
    inputs.requires_grad_(True)
    dR = derivative_wrt_rho(model, inputs)
    d2Rdx2 = torch.autograd.grad(dR, inputs, grad_outputs=torch.ones_like(dR),
                                  create_graph=True, retain_graph=True)[0]
    return d2Rdx2[:, 0:1]


# ======================================
# 4. Puntos de anclaje optimizados
# ======================================
ANCHOR_POINTS = {
    1: [0.03, 0.06],
    2: [0.04, 0.07],
    3: [0.025, 0.05, 0.075]
}


# ======================================
# 5. Loss function adaptativa por fase
# ======================================

def loss_function(model, rho_col, n_col, active_orders, phase_name=''):
    """
    active_orders: lista de órdenes activos en esta fase
    """
    inputs = torch.cat([rho_col, n_col], dim=1)
    inputs.requires_grad_(True)
    
    R_pred = model(inputs)
    dR = derivative_wrt_rho(model, inputs)
    d2R = second_derivative_wrt_rho(model, inputs)
    
    residual = (rho_col**2)*d2R + rho_col*dR + (rho_col**2 - n_col**2)*R_pred
    
    # Solo calcular pérdida para órdenes activos
    loss_pde_total = 0.0
    for n_value in active_orders:
        mask = (n_col.squeeze() == n_value)
        if torch.any(mask):
            weight = LOSS_WEIGHTS[n_value]
            loss_pde_total += weight * torch.mean(residual[mask]**2)
    
    if len(active_orders) > 0:
        loss_pde = loss_pde_total / len(active_orders)
    else:
        loss_pde = 0.0
    
    # BCs solo para órdenes activos
    eps = 1e-5
    loss_bc_total = 0.0
    
    for n_value in active_orders:
        rho_bc0 = torch.tensor([[eps]], dtype=torch.float32, device=device)
        n_bc0 = torch.tensor([[n_value]], dtype=torch.float32, device=device)
        inputs_bc0 = torch.cat([rho_bc0, n_bc0], dim=1)
        inputs_bc0.requires_grad_(True)
        
        R_bc0 = model(inputs_bc0)
        dR_bc0 = derivative_wrt_rho(model, inputs_bc0)
        
        if n_value == 0:
            loss_bc0 = (R_bc0 - 1.0)**2 + (dR_bc0 - 0.0)**2
            loss_bc_total += 8.0 * torch.mean(loss_bc0)
            
        elif n_value == 1:
            loss_bc1 = (R_bc0 - 0.0)**2 + (dR_bc0 - 0.5)**2
            loss_bc_total += 12.0 * torch.mean(loss_bc1)
            
            for rho_val in ANCHOR_POINTS[1]:
                rho_bc = torch.tensor([[rho_val]], dtype=torch.float32, device=device)
                n_bc = torch.tensor([[n_value]], dtype=torch.float32, device=device)
                inputs_bc = torch.cat([rho_bc, n_bc], dim=1)
                inputs_bc.requires_grad_(True)
                
                J1_exact = jv(1, rho_val)
                J0_exact = jv(0, rho_val)
                J2_exact = jv(2, rho_val)
                J1_der = 0.5*(J0_exact - J2_exact)
                
                R_bc = model(inputs_bc)
                dR_bc = derivative_wrt_rho(model, inputs_bc)
                
                R_t = torch.tensor([[J1_exact]], dtype=torch.float32, device=device)
                dR_t = torch.tensor([[J1_der]], dtype=torch.float32, device=device)
                
                loss_anchor = (R_bc - R_t)**2 + (dR_bc - dR_t)**2
                loss_bc_total += 18.0 * torch.mean(loss_anchor)
            
        elif n_value >= 2:
            d2R_bc0 = second_derivative_wrt_rho(model, inputs_bc0)
            
            if n_value == 2:
                loss_bc2 = (R_bc0 - 0.0)**2 + (d2R_bc0 - 0.25)**2
                loss_bc_total += 12.0 * torch.mean(loss_bc2)
                
                for rho_val in ANCHOR_POINTS[2]:
                    rho_bc = torch.tensor([[rho_val]], dtype=torch.float32, device=device)
                    n_bc = torch.tensor([[n_value]], dtype=torch.float32, device=device)
                    inputs_bc = torch.cat([rho_bc, n_bc], dim=1)
                    inputs_bc.requires_grad_(True)
                    
                    Jn_exact = jv(n_value, rho_val)
                    Jn_m1 = jv(n_value-1, rho_val)
                    Jn_p1 = jv(n_value+1, rho_val)
                    Jn_der = 0.5*(Jn_m1 - Jn_p1)
                    
                    R_bc = model(inputs_bc)
                    dR_bc = derivative_wrt_rho(model, inputs_bc)
                    
                    R_t = torch.tensor([[Jn_exact]], dtype=torch.float32, device=device)
                    dR_t = torch.tensor([[Jn_der]], dtype=torch.float32, device=device)
                    
                    loss_anchor = (R_bc - R_t)**2 + (dR_bc - dR_t)**2
                    loss_bc_total += 22.0 * torch.mean(loss_anchor)
                
            elif n_value == 3:
                loss_bc3 = (R_bc0 - 0.0)**2 + (dR_bc0 - 0.0)**2
                loss_bc_total += 15.0 * torch.mean(loss_bc3)
                
                for rho_val in ANCHOR_POINTS[3]:
                    rho_bc = torch.tensor([[rho_val]], dtype=torch.float32, device=device)
                    n_bc = torch.tensor([[n_value]], dtype=torch.float32, device=device)
                    inputs_bc = torch.cat([rho_bc, n_bc], dim=1)
                    inputs_bc.requires_grad_(True)
                    
                    J3_exact = jv(3, rho_val)
                    J2_exact = jv(2, rho_val)
                    J4_exact = jv(4, rho_val)
                    J3_der = 0.5*(J2_exact - J4_exact)
                    
                    R_bc = model(inputs_bc)
                    dR_bc = derivative_wrt_rho(model, inputs_bc)
                    
                    R_t = torch.tensor([[J3_exact]], dtype=torch.float32, device=device)
                    dR_t = torch.tensor([[J3_der]], dtype=torch.float32, device=device)
                    
                    loss_anchor = (R_bc - R_t)**2 + (dR_bc - dR_t)**2
                    loss_bc_total += 28.0 * torch.mean(loss_anchor)
    
    return loss_pde + loss_bc_total


# ======================================
# 6. Generación de datos
# ======================================

def generate_collocation_points(N_points, alpha=4.5):
    eps_np = 1e-5
    t = np.linspace(0.0, 1.0, N_points)
    rho_col_np = eps_np + (rho_max - eps_np) * (t ** alpha)
    return rho_col_np

print("\n📊 Generando datos de entrenamiento...")
rho_all = []
n_all = []

for n_value in N_ORDERS:
    rho_np = generate_collocation_points(N_col_per_order, alpha=alpha_power)
    rho_tensor = torch.tensor(rho_np, dtype=torch.float32).view(-1, 1)
    n_tensor = torch.full_like(rho_tensor, n_value)
    
    rho_all.append(rho_tensor)
    n_all.append(n_tensor)

rho_train = torch.cat(rho_all, dim=0).to(device)
n_train = torch.cat(n_all, dim=0).to(device)

print(f"   Total: {len(rho_train)} puntos ({N_col_per_order} por orden)")


# ======================================
# 7. ENTRENAMIENTO POR FASES (CURRICULUM)
# ======================================

optimizer = optim.Adam(model.parameters(), lr=lr_initial)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, 
                                                   patience=800, verbose=False, min_lr=1e-6)

loss_history = []
phase_markers = []  # Para marcar cambios de fase en la gráfica

print("\n" + "="*70)
print(" "*15 + "🎓 ENTRENAMIENTO POR CURRICULUM LEARNING")
print("="*70)

total_start = time.time()
global_epoch = 0

# FASE 1: Solo J0 (establecer base sólida)
print(f"\n{'='*70}")
print(f"📚 FASE 1: Solo J0 - {EPOCHS_PER_PHASE['phase_1_J0']} épocas")
print(f"{'='*70}")

phase_markers.append(global_epoch)
active_orders = [0]
mask_orders = torch.isin(n_train, torch.tensor(active_orders, device=device))
rho_phase = rho_train[mask_orders]
n_phase = n_train[mask_orders]

for epoch in range(EPOCHS_PER_PHASE['phase_1_J0']):
    optimizer.zero_grad()
    
    indices = torch.randperm(len(rho_phase))
    loss = loss_function(model, rho_phase[indices], n_phase[indices], active_orders, 'phase_1')
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss.item())
    
    loss_history.append(loss.item())
    global_epoch += 1
    
    if epoch % 1000 == 0:
        print(f"  Época {epoch:5d}, Pérdida: {loss.item():.9f}")

print(f"✅ Fase 1 completada")

# FASE 2: J0 + J1
print(f"\n{'='*70}")
print(f"📚 FASE 2: J0 + J1 - {EPOCHS_PER_PHASE['phase_2_J0J1']} épocas")
print(f"{'='*70}")

phase_markers.append(global_epoch)
active_orders = [0, 1]
mask_orders = torch.isin(n_train, torch.tensor(active_orders, device=device))
rho_phase = rho_train[mask_orders]
n_phase = n_train[mask_orders]

for epoch in range(EPOCHS_PER_PHASE['phase_2_J0J1']):
    optimizer.zero_grad()
    
    indices = torch.randperm(len(rho_phase))
    loss = loss_function(model, rho_phase[indices], n_phase[indices], active_orders, 'phase_2')
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss.item())
    
    loss_history.append(loss.item())
    global_epoch += 1
    
    if epoch % 1000 == 0:
        print(f"  Época {epoch:5d}, Pérdida: {loss.item():.9f}")

print(f"✅ Fase 2 completada")

# FASE 3: J0 + J1 + J2
print(f"\n{'='*70}")
print(f"📚 FASE 3: J0 + J1 + J2 - {EPOCHS_PER_PHASE['phase_3_J0J1J2']} épocas")
print(f"{'='*70}")

phase_markers.append(global_epoch)
active_orders = [0, 1, 2]
mask_orders = torch.isin(n_train, torch.tensor(active_orders, device=device))
rho_phase = rho_train[mask_orders]
n_phase = n_train[mask_orders]

for epoch in range(EPOCHS_PER_PHASE['phase_3_J0J1J2']):
    optimizer.zero_grad()
    
    indices = torch.randperm(len(rho_phase))
    loss = loss_function(model, rho_phase[indices], n_phase[indices], active_orders, 'phase_3')
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss.item())
    
    loss_history.append(loss.item())
    global_epoch += 1
    
    if epoch % 1000 == 0:
        print(f"  Época {epoch:5d}, Pérdida: {loss.item():.9f}")

print(f"✅ Fase 3 completada")

# FASE 4: Todos juntos
print(f"\n{'='*70}")
print(f"📚 FASE 4: J0 + J1 + J2 + J3 - {EPOCHS_PER_PHASE['phase_4_all']} épocas")
print(f"{'='*70}")

phase_markers.append(global_epoch)
active_orders = [0, 1, 2, 3]

for epoch in range(EPOCHS_PER_PHASE['phase_4_all']):
    optimizer.zero_grad()
    
    indices = torch.randperm(len(rho_train))
    loss = loss_function(model, rho_train[indices], n_train[indices], active_orders, 'phase_4')
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss.item())
    
    loss_history.append(loss.item())
    global_epoch += 1
    
    if epoch % 1000 == 0:
        print(f"  Época {epoch:5d}, Pérdida: {loss.item():.9f}")

print(f"✅ Fase 4 completada")

# FASE 5: Fine-tuning final con LR bajo
print(f"\n{'='*70}")
print(f"📚 FASE 5: Fine-tuning final - {EPOCHS_PER_PHASE['phase_5_finetune']} épocas")
print(f"{'='*70}")

phase_markers.append(global_epoch)
# Reducir LR manualmente para fine-tuning
for param_group in optimizer.param_groups:
    param_group['lr'] = 0.0005

for epoch in range(EPOCHS_PER_PHASE['phase_5_finetune']):
    optimizer.zero_grad()
    
    indices = torch.randperm(len(rho_train))
    loss = loss_function(model, rho_train[indices], n_train[indices], active_orders, 'phase_5')
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    
    loss_history.append(loss.item())
    global_epoch += 1
    
    if epoch % 1000 == 0:
        print(f"  Época {epoch:5d}, Pérdida: {loss.item():.9f}")

print(f"✅ Fase 5 completada")

total_time = time.time() - total_start
print(f"\n{'='*70}")
print(f"✅ ENTRENAMIENTO COMPLETO en {total_time:.2f} segundos")
print(f"   Total de épocas: {global_epoch}")
print(f"{'='*70}")


# ======================================
# 8. Evaluación
# ======================================

model.eval()
rho_test_np = np.linspace(0, rho_max, 300)

print("\n" + "="*70)
print(" "*15 + "EVALUACIÓN DE CADA ORDEN DE BESSEL")
print("="*70)

results = {}

for n_value in N_ORDERS:
    print(f"\n{'='*70}")
    print(f" "*25 + f"ORDEN n = {n_value}")
    print(f"{'='*70}")
    
    rho_test = torch.tensor(rho_test_np, dtype=torch.float32).view(-1, 1).to(device)
    n_test = torch.full_like(rho_test, n_value)
    inputs_test = torch.cat([rho_test, n_test], dim=1)
    
    start_inf = time.time()
    with torch.no_grad():
        R_pred = model(inputs_test).cpu().numpy().flatten()
    end_inf = time.time()
    inference_time = end_inf - start_inf
    
    R_exact = jv(n_value, rho_test_np)
    
    mse = np.mean((R_pred - R_exact)**2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(R_pred - R_exact))
    mean_abs_error = np.mean(np.abs(R_pred - R_exact))
    
    threshold = 0.1
    mask_robust = np.abs(R_exact) > threshold
    if np.any(mask_robust):
        relative_error_robust = np.abs(R_pred[mask_robust] - R_exact[mask_robust]) / np.abs(R_exact[mask_robust])
        mean_relative_error_robust = np.mean(relative_error_robust)
        similarity_robust = (1 - mean_relative_error_robust) * 100
    else:
        similarity_robust = 0.0
    
    ss_res = np.sum((R_exact - R_pred)**2)
    ss_tot = np.sum((R_exact - np.mean(R_exact))**2)
    r2_score = 1 - (ss_res / ss_tot)
    
    correlation = np.corrcoef(R_exact, R_pred)[0, 1]
    
    results[n_value] = {
        'R_pred': R_pred,
        'R_exact': R_exact,
        'similarity_robust': similarity_robust,
        'rmse': rmse,
        'max_error': max_error,
        'r2': r2_score,
        'correlation': correlation,
        'inference_time': inference_time
    }
    
    print(f"MSE (Error Cuadrático Medio):              {mse:.8f}")
    print(f"RMSE (Raíz del MSE):                       {rmse:.8f}")
    print(f"Error Máximo:                              {max_error:.8f}")
    print(f"Error Absoluto Promedio:                   {mean_abs_error:.8f}")
    print("-"*70)
    print(f"SIMILITUD ROBUSTA (|exact|>0.1):           {similarity_robust:.6f}%  ⭐")
    print(f"R² (Coeficiente de Determinación):         {r2_score:.8f}")
    print(f"Correlación de Pearson:                    {correlation:.8f}")
    print(f"Tiempo de inferencia:                      {inference_time:.6f} s")

print("\n" + "="*70)
print(" "*20 + "RESUMEN COMPARATIVO")
print("="*70)
print(f"Tiempo total de entrenamiento:             {total_time:.4f} segundos")
print(f"\nSimilitud Robusta por orden:")
for n_value in N_ORDERS:
    sim = results[n_value]['similarity_robust']
    emoji = '✅' if sim > 98 else ('🟡' if sim > 95 else '⚠️')
    print(f"  J{n_value}: {sim:.4f}%  {emoji}")


# ======================================
# 9. Visualizaciones
# ======================================

# Figura 1: Comparación
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, n_value in enumerate(N_ORDERS):
    ax = axes[idx]
    res = results[n_value]
    
    ax.plot(rho_test_np, res['R_exact'], label=f"J{n_value} Analítico", 
            linewidth=2.5, color='black')
    ax.plot(rho_test_np, res['R_pred'], '--', 
            label=f"PINN v3 (Sim: {res['similarity_robust']:.2f}%)", 
            linewidth=2, color='blue')
    
    ax.set_xlabel("ρ", fontsize=12)
    ax.set_ylabel(f"J{n_value}(ρ)", fontsize=12)
    ax.set_title(f"Función de Bessel de orden {n_value}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.6, 1.2)

plt.tight_layout()
plt.savefig('Resultados/bessel_general_v3_comparacion.png', dpi=300, bbox_inches='tight')
plt.show()

# Figura 2: Errores
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, n_value in enumerate(N_ORDERS):
    ax = axes[idx]
    res = results[n_value]
    error = np.abs(res['R_pred'] - res['R_exact'])
    
    ax.plot(rho_test_np, error, linewidth=2, color='red')
    ax.set_xlabel("ρ", fontsize=12)
    ax.set_ylabel("|Error|", fontsize=12)
    ax.set_title(f"Error Absoluto - J{n_value} (Max: {res['max_error']:.6f})", 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('Resultados/bessel_general_v3_errores.png', dpi=300, bbox_inches='tight')
plt.show()

# Figura 3: Loss con marcadores de fases
plt.figure(figsize=(14, 6))
plt.plot(loss_history, linewidth=1.5, color='darkblue', alpha=0.7)

# Marcar cambios de fase
phase_names = ['Fase 1\n(J0)', 'Fase 2\n(J0+J1)', 'Fase 3\n(J0+J1+J2)', 
               'Fase 4\n(Todos)', 'Fase 5\n(Fine-tune)']
colors_phases = ['red', 'orange', 'green', 'purple', 'brown']

for i, marker in enumerate(phase_markers):
    plt.axvline(x=marker, color=colors_phases[i], linestyle='--', alpha=0.7, linewidth=2)
    plt.text(marker + 200, max(loss_history)*0.5, phase_names[i], 
             rotation=0, fontsize=9, color=colors_phases[i], fontweight='bold')

plt.xlabel("Época Global", fontsize=12)
plt.ylabel("Pérdida", fontsize=12)
plt.title("Evolución de la pérdida - Curriculum Learning\n(5 fases de entrenamiento progresivo)", 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('Resultados/bessel_general_v3_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# Figura 4: Familia con porcentajes
plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'purple']

for idx, n_value in enumerate(N_ORDERS):
    res = results[n_value]
    sim_pct = res['similarity_robust']
    
    plt.plot(rho_test_np, res['R_exact'], '-', 
             label=f"J{n_value} Analítico", 
             linewidth=2.5, color=colors[idx], alpha=0.6)
    
    plt.plot(rho_test_np, res['R_pred'], '--', 
             label=f"J{n_value} PINN ({sim_pct:.2f}%)", 
             linewidth=2, color=colors[idx], alpha=0.9)

plt.xlabel("ρ", fontsize=12)
plt.ylabel("Jn(ρ)", fontsize=12)
plt.title("Familia de Funciones de Bessel - v3 Curriculum Learning\n(Líneas continuas: analíticas, discontinuas: PINN con % similitud)", 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=9, ncol=2, loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(0, rho_max)
plt.ylim(-0.5, 1.1)
plt.tight_layout()
plt.savefig('Resultados/bessel_general_v3_familia.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Visualizaciones guardadas en 'Resultados/'")
print("\n🎉 ¡Red v3 con CURRICULUM LEARNING completada!")
print("\n📝 ESTRATEGIA:")
print("   Fase 1: Solo J0 (5000 épocas)")
print("   Fase 2: J0 + J1 (6000 épocas)")
print("   Fase 3: J0 + J1 + J2 (8000 épocas)")
print("   Fase 4: Todos (12000 épocas)")
print("   Fase 5: Fine-tuning (4000 épocas)")
print(f"   TOTAL: {sum(EPOCHS_PER_PHASE.values())} épocas")

