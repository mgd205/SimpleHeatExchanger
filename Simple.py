import streamlit as st
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from seaborn.palettes import blend_palette

st.set_page_config(layout="wide")

# Explaining the simulator
st.sidebar.title('About the Simulator:')
st.sidebar.write('This is a simulator of a simple tubular heat exchanger that heats a fluid as it passes through it. When running the simulation, you will be able to observe the temperature profile of the fluid throughout the exchanger as time passes. You will also be able to view steady-state temperatures along the length of the exchanger.')
st.sidebar.write('Below is an image exemplifying this exchanger, created by myself.')
st.sidebar.image('Simple tubular img #1.png', use_column_width=True)
st.sidebar.write('An application for this case is the coils used in passing heating systems. The coils consist of tubes or tube systems through which fluids pass and are heated by an external heat source.')
st.sidebar.write('This case can also represent any industrial piping used to heat fluids through an external heat source.')
st.sidebar.write('This simulator uses the following energy balance equation for the fluid that passes through the exchanger, considering the principle of energy conservation:')
st.sidebar.image('Simple tubular equations.jpg', use_column_width=True)

def run_simulation(L, r, n, m, Cp, rho, Ti, T0, q_fluxo, t_final, dt):
    dx = L / n
    x = np.linspace(dx/2, L-dx/2, n)
    T = np.ones(n) * T0
    t = np.arange(0, t_final, dt)

    # Creating the figure for the steady-state graph
    fig_permanente = plt.figure(figsize=(8, 6))

    # Function that defines the ODE for temperature variation
    def dTdt_function(T, t):
        dTdt = np.zeros(n)
        dTdt[1:n] = (m*Cp*(T[0:n-1]-T[1:n])+q_fluxo*2*np.pi*r*dx)/(rho*Cp*dx*np.pi*r**2)
        dTdt[0] = (m*Cp*(Ti-T[0])+q_fluxo*2*np.pi*r*dx)/(rho*Cp*dx*np.pi*r**2)
        return dTdt

    # Solving the ODE using odeint
    T_out = odeint(dTdt_function, T, t)
    T_out = T_out

    # Creating the DataFrame
    df_Temp = pd.DataFrame(np.array(T_out), columns=x)

    # Creating a Color Palette
    paleta_calor = blend_palette(['yellow', 'orange','red'], as_cmap=True, n_colors=100)

    # Function that updates the plot
    def update_plot(t):
        plt.clf()
        line = pd.DataFrame(df_Temp.iloc[t, :]).T
        sns.heatmap(line, cmap=paleta_calor)
        plt.title(f'Time: {t} (s)')
        plt.gca().set_xticklabels(['{:.2f}'.format(val) for val in x])

    # Creating figure for the animation
    fig_animacao = plt.figure(figsize=(8, 6))

    # Creating the animation
    ani = FuncAnimation(fig_animacao, update_plot, frames=df_Temp.shape[0], repeat=False)

    # Saving the animation as a gif
    save = ani.save('Temperature Variation - Case I.gif', writer='pillow', fps=10)

    # Displaying the simulation
    with st.expander("Visualization of the real-time Simulation of the Fluid (Click here to see)"):
        st.write('Variation in the temperature of the fluid passing through the exchanger over time and length.')
        st.write('Time represented above the GIF, in seconds. Temperatures in Kelvin on the y-axis. Length of the exchanger in meters on the x-axis of the GIF.')
        st.image('Temperature Variation - Case I.gif')

    # Displaying the graph of temperature variation along length in steady-state
    plt.figure(fig_permanente)
    plt.plot(x, df_Temp.iloc[-1, :], color='blue')
    plt.xlabel('Length (m)')
    plt.ylabel('Temperature (K)')
    plt.title('Fluid temperature along the length of the exchanger at steady-state.')
    st.pyplot(plt)

st.title('TROCAL Simulator - Simulation of a Simple Tubular Heat Exchanger')

col1, col2 = st.columns(2)
with col1:
  st.header('Parameters')
  st.write('ATTENTION: In the "Results" section, you will find a button that runs the simulation with a pre-defined example ("Run standard example"). This example takes around 30 seconds to run, depending on your connection speed. If you want to use your own input values, use the "Run simulation" button. It is recommended to use a number of nodes between 10 and 30, depending on the specific example used.')
  # Input Values
  L = st.number_input('Length of the tube (m)', min_value=0.0)
  r = st.number_input('Radius of the tube (m)', min_value=0.0)
  n = st.number_input('Number of nodes for discretization', min_value=1)
  m = st.number_input('Mass flow (kg/s)', min_value=0.0)
  Cp = st.number_input('Specific heat capacity of the fluid (J/kg.K)', min_value=0.0)
  rho = st.number_input('Specific mass of the fluid (kg/m³)', min_value=0.0)
  Ti = st.number_input('Inlet temperature of the fluid (K)')
  T0 = st.number_input('Initial temperature of the exchanger (K)')
  q_fluxo = st.number_input('Heat flux (W/m²)', min_value=0.0)
  t_final = st.number_input('Simulation time (s)', min_value=0.0)
  dt = st.number_input('Time step (s)', min_value=0.0)

with col2:
  st.header('Results')
  if st.button('Run simulation'):
      run_simulation(L, r, n, m, Cp, rho, Ti, T0, q_fluxo, t_final, dt)
  elif st.button('Run standard example'):
      run_simulation(10, 0.1, 10, 3, 4180, 995.61, 400, 300, 10000, 210, 1)
