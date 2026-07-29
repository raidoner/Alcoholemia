import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador de Alcoholemia", layout="centered")

st.title("🍺 Simulador de Alcoholemia (BAC)")
st.write("Calcula tu curva de alcohol en sangre con cinética de absorción y eliminación.")

# --- BARRA LATERAL / CONTROLES ---
st.sidebar.header("Parámetros del consumo")

peso = st.sidebar.number_input("Peso (kg)", min_value=40, max_value=150, value=75)
sexo = st.sidebar.radio("Sexo", ["Hombre", "Mujer"])
r_widmark = 0.68 if sexo == "Hombre" else 0.55

estomago = st.sidebar.selectbox(
    "Estado del estómago (Absorción)",
    ["En ayunas (Rápida)", "Comida ligera", "Cena normal", "Cena copiosa", "Fast food / Grasas (Lenta)"]
)

# Mapeo de ka según la comida
ka_dict = {
    "En ayunas (Rápida)": 2.5,
    "Comida ligera": 1.5,
    "Cena normal": 0.9,
    "Cena copiosa": 0.6,
    "Fast food / Grasas (Lenta)": 0.4
}
ka = ka_dict[estomago]

duracion = st.sidebar.slider("Duración del consumo (horas)", 1, 24, 10)
tasa = st.sidebar.number_input("Tasa de ingesta (g/hora)", min_value=0.5, max_value=100.0, value=15.0, step=0.5)
tiempo_sim = st.sidebar.slider("Horas totales a simular", 5, 48, 10)

# --- SECCIÓN: CONSULTA EN UN INSTANTE PUNTUAL ---
st.sidebar.markdown("---")
st.sidebar.header("🔍 Consultar instante específico")
consultar_hora = st.sidebar.checkbox("Activar consulta de hora")

if consultar_hora:
    t_consulta = st.sidebar.number_input(
        "Hora a consultar (h)", 
        min_value=0.0, 
        max_value=float(tiempo_sim), 
        value=min(2.0, float(tiempo_sim)), 
        step=0.5
    )

# --- SECCIÓN: CONDICIONES INICIALES (t=0) ---
st.sidebar.markdown("---")
st.sidebar.header("🚀 Condiciones Iniciales (t=0)")
A0 = st.sidebar.number_input("Alcohol inicial en estómago (g)", min_value=0.0, value=0.0, step=1.0)
C0 = st.sidebar.number_input("BAC inicial en sangre (g/L)", min_value=0.0, value=0.0, step=0.05)
Acum0 = st.sidebar.number_input("Alcohol acumulado previo (g)", min_value=0.0, value=0.0, step=1.0)

# --- CÁLCULOS FARMACOCINÉTICOS ---
Vd = peso * r_widmark
Vmax = 0.18
Km = 0.08

def sistema_edo(t, y):
    A, C, Acum = y
    R_in = tasa if t <= duracion else 0.0
    C_pos = max(0.0, C)
    
    dA_dt = R_in - ka * A
    dC_dt = (ka * A) / Vd - (Vmax * C_pos) / (Km + C_pos)
    dAcum_dt = ka * A
    return [dA_dt, dC_dt, dAcum_dt]

t_eval = np.linspace(0, tiempo_sim, 1000)

# dense_output=True permite evaluar el sistema en cualquier punto continuo t
sol = solve_ivp(
    sistema_edo, 
    [0, tiempo_sim], 
    [A0, C0, Acum0], 
    t_eval=t_eval, 
    method='RK45', 
    dense_output=True
)

t = sol.t
BAC = sol.y[1]
Absorbido = sol.y[2]

# Métricas principales
idx_max = np.argmax(BAC)
Cmax = BAC[idx_max]
Tmax = t[idx_max]

indices_cero = np.where((t > Tmax) & (BAC <= 0.005))[0]
T_cero = t[indices_cero[0]] if len(indices_cero) > 0 else None

# --- MOSTRAR RESULTADOS GENERALES ---
col1, col2, col3 = st.columns(3)
col1.metric("Pico Máximo (BAC)", f"{Cmax:.2f} g/L")
col2.metric("Hora del Pico", f"{Tmax:.1f} h")
col3.metric("Alcohol a 0", f"{T_cero:.1f} h" if T_cero else "> simulación")

# --- MOSTRAR CONSULTA ESPECÍFICA (SI ESTÁ ACTIVADA) ---
if consultar_hora:
    y_instante = sol.sol(t_consulta)  # Evalúa la EDO exactamente en t_consulta
    A_inst = max(0.0, y_instante[0])
    C_inst = max(0.0, y_instante[1])
    Acum_inst = max(0.0, y_instante[2])
    
    st.markdown("---")
    st.subheader(f"⏱️ Estado a las {t_consulta:.1f} horas")
    q1, q2, q3 = st.columns(3)
    q1.metric("En Estómago (GI)", f"{A_inst:.1f} g")
    q2.metric("BAC (Sangre)", f"{C_inst:.3f} g/L")
    q3.metric("Total Absorbido", f"{Acum_inst:.1f} g")

# --- GRÁFICA ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

# Gráfica BAC
max_ylim = max(Cmax * 1.2, 0.6)
ax1.plot(t, BAC, 'k-', linewidth=2.5, label='BAC (g/L)')
ax1.axvspan(0, duracion, color='#e6cc33', alpha=0.25, label='Tiempo bebiendo')
ax1.axhline(y=0.5, color='r', linestyle='--', label='Límite 0.5 g/L')
ax1.plot(Tmax, Cmax, 'ro')

if T_cero:
    ax1.plot(T_cero, 0, 'go')

# Marcar instante consultado en la gráfica si aplica
if consultar_hora:
    ax1.plot(t_consulta, C_inst, 'bo', markersize=8, label=f'Consulta ({t_consulta:.1f}h)')

ax1.set_title("Evolución de Alcoholemia")
ax1.set_ylabel("g/L")
ax1.set_ylim(0, max_ylim)
ax1.grid(True)
ax1.legend(loc='upper right')

# Gráfica Wagner-Nelson
total_ingerido = Acum0 + tasa * np.minimum(t, duracion)
ax2.plot(t, total_ingerido, 'k--', label='Ingerido (g)')
ax2.plot(t, Absorbido, 'b-', label='Absorbido (g)')

if consultar_hora:
    ax2.plot(t_consulta, Acum_inst, 'bo', markersize=8)

ax2.set_title("Acumulado de Alcohol")
ax2.set_xlabel("Horas")
ax2.set_ylabel("Gramos")
ax2.grid(True)
ax2.legend(loc='lower right')

st.pyplot(fig)
