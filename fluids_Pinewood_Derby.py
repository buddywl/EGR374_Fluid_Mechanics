import matplotlib.pyplot as plt
import numpy as np
import math


# =============================
# SIMULATION FUNCTION
# =============================

# CONSTANTS
g = 9.81
rho = 1.2
mu_air = 1.8e-5   # dynamic viscosity of air

# Wheel parameters
r = 0.015
I = 0.5 * 0.03 * r**2

# Track
theta = np.radians(5)
track_length = 1.5        # meters

# Characteristic length (car height-ish)
L = 0.044      # meters

def compute_Re_data(v_data):
    return rho * v_data * L / mu_air

F1_v = np.array([1.8, 3.5, 6.8, 11])
F1_cd = np.array([0.4188, 0.1265, 0.7918, 0.6743])

Block_v = np.array([1.8, 3.5, 6.8, 11])
Block_cd = np.array([0.5252, 0.3425, 0.9630, 0.7902])

Champion_v = np.array([1.8, 3.5, 6.8, 11])
Champion_cd = np.array([0, 0.0025, 0.9046, 0.6830])

F1_Re = compute_Re_data(F1_v)
Block_Re = compute_Re_data(Block_v)
Champion_Re = compute_Re_data(Champion_v)

def simulate_car(m, A, mu_roll, label, Re_data, cd_data):

    # Time setup
    step = 0.001
    time = 1.8
    sec = np.arange(0, time, step)

    acc = np.zeros(sec.size)
    vel = np.zeros(sec.size)
    pos = np.zeros(sec.size)

    def Cd_from_Re(v, Re_data, cd_data):
        Re = rho * abs(v) * L / mu_air
        if Re < 1:
            Re = 1
        return np.interp(Re, Re_data, cd_data)

    PE = np.zeros(sec.size)
    KE = np.zeros(sec.size)
    KE_rot = np.zeros(sec.size)
    E_total = np.zeros(sec.size)

    # Height decreases as we go down ramp
    h0 = track_length * np.sin(theta)
    def height(x):
        if x < track_length:
            return h0 - x * np.sin(theta)
        else:
            return 0.0

    # Set initial energy
    PE[0] = m * g * h0
    KE[0] = 0.0
    KE_rot[0] = 0.0
    E_total[0] = PE[0]

    # Euler integration
    for i in range(sec.size - 1):

        if pos[i] < track_length:
            angle = theta
        else:
            angle = 0

        # Forces
        gravity = g * np.sin(angle)
        rolling = mu_roll * g * np.cos(angle)

        Cd = Cd_from_Re(vel[i], Re_data, cd_data)
        drag = (rho * Cd * A * vel[i] * abs(vel[i])) / (2 * m)

        acc[i] = (gravity - rolling - drag) / (1 + I / (m * r**2))
        vel[i+1] = vel[i] + acc[i] * step
        pos[i+1] = pos[i] + vel[i] * step

        h = height(pos[i+1])

        PE[i+1] = m * g * h
        KE[i+1] = 0.5 * m * vel[i+1]**2
        omega = vel[i+1] / r
        KE_rot[i+1] = 0.5 * I * omega**2
        E_total[i] = PE[i] + KE[i] + KE_rot[i]

        if pos[i] >= track_length * 2:
            vel[i+1] = 0
            pos[i+1] = track_length * 2
            acc[i+1] = 0
            continue

    # Energy loss vs initial total energy
    E_loss = E_total[0] - E_total

    return sec, pos, vel, acc, PE, KE, KE_rot, E_total, E_loss, label


# =============================
# DEFINE CARS
# -----------------------------
# place car on flat surface
# increase angle of surface until car rolls
# mu_roll = tan(angle_in_degrees converted to radians)
# e.g. math.tan(math.radians(2.2)) for a 2.2-degree rolling angle
# =============================

BlockCar = simulate_car(
    m=0.15,
    A=0.0020,
    mu_roll=math.tan(math.radians(2.2)),
    label="Block Car",
    Re_data=Block_Re,
    cd_data=Block_cd
)

F1Car = simulate_car(
    m=0.15,
    A=0.0018,
    mu_roll=math.tan(math.radians(2.4)),
    label="F1 Car",
    Re_data=F1_Re,
    cd_data=F1_cd
)

Champion = simulate_car(
    m=0.15,
    A=0.0014,
    mu_roll=math.tan(math.radians(2.95)),
    label="Championship Car",
    Re_data=Champion_Re,
    cd_data=Champion_cd
)


# =============================
# PLOTTING
# =============================
# POSITION, VELOCITY, ACCELERATION
color_map = {
    "Block Car": "dodgerblue",
    "F1 Car": "tomato",
    "Championship Car": "forestgreen"
}

fig = plt.figure(figsize=(10, 10))
gridsize = (3, 1)

ax1 = plt.subplot2grid(gridsize, (0, 0))
ax2 = plt.subplot2grid(gridsize, (1, 0))
ax3 = plt.subplot2grid(gridsize, (2, 0))

cars = [BlockCar, F1Car, Champion]

for sec, pos, vel, acc, PE, KE, KE_rot, E_total, E_loss, label in cars:
    color = color_map[label]
    ax1.plot(sec, pos, lw=2, label=label, color=color)
    ax2.plot(sec, vel, lw=2, label=label, color=color)
    ax3.plot(sec, acc, lw=2, label=label, color=color)

ax1.set_title('PINEWOOD DERBY MODEL', fontsize=14)
ax1.set_ylabel('Position [m]')
ax2.set_ylabel('Velocity [m/s]')
ax3.set_ylabel('Acceleration [m/s²]')
ax3.set_xlabel('Time [s]')

for ax in [ax1, ax2, ax3]:
    ax.grid(ls='-.', lw=0.5)
    ax.legend()

plt.tight_layout()

# ENERGY PLOTS
figE, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

E_min = float('inf')
E_max = float('-inf')

for sec, pos, vel, acc, PE, KE, KE_rot, E_total, E_loss, label in cars:
    E_min = min(E_min, np.min(E_total))
    E_max = max(E_max, np.max(E_total))

padding = 0.05 * (E_max - E_min)
E_min -= padding
E_max += padding

for ax, (sec, pos, vel, acc, PE, KE, KE_rot, E_total, E_loss, label) in zip(axes, cars):
    color = color_map[label]

    ax.plot(sec, PE,      '--', color=color, label='PE')
    ax.plot(sec, KE,      color=color, alpha=0.7, label='KE (trans)')
    ax.plot(sec, KE_rot,  ':', color=color, label='KE (rot)')
    ax.plot(sec, E_total, lw=2, color=color, label='Total')
    ax.plot(sec, E_loss,  lw=1.5, color=color, alpha=0.5, linestyle='dashdot', label='Energy lost')

    ax.set_title(label)
    ax.set_ylabel('Energy [J]')
    ax.set_ylim(E_min, E_max)
    ax.grid(ls='-.', lw=0.5)
    ax.legend(fontsize=8)

axes[-1].set_xlabel('Time [s]')

plt.tight_layout()
plt.savefig('output\\kinematics.png', dpi=150)
figE.savefig('output\\energy.png', dpi=150)
plt.show()
print("Done.")
