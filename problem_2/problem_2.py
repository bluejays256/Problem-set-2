import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
print("Part a")
# Stereographic projection function
def stereographic_projection(x, y, z):
    return x / (1 - z), y / (1 - z)

# Define curves on the sphere
theta = np.linspace(0, np.pi, 100)  # Polar angle
phi1 = np.pi / 4  # First curve
phi2 = np.pi / 2  # Second curve

# First curve: meridian (constant phi)
x1 = np.sin(theta) * np.cos(phi1)
y1 = np.sin(theta) * np.sin(phi1)
z1 = np.cos(theta)

# Second curve: another meridian (constant phi)
x2 = np.sin(theta) * np.cos(phi2)
y2 = np.sin(theta) * np.sin(phi2)
z2 = np.cos(theta)

# Project curves onto the plane
x1_proj, y1_proj = stereographic_projection(x1, y1, z1)
x2_proj, y2_proj = stereographic_projection(x2, y2, z2)

# Create figure
fig = plt.figure(figsize=(12, 6))

# ---- First plot: Sphere with intersecting curves ----
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x1, y1, z1, 'b', label='Curve 1')
ax1.plot(x2, y2, z2, 'r', label='Curve 2')
ax1.scatter([0], [0], [1], color='black', s=50, label='Intersection')

# Sphere surface
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(X, Y, Z, color='gray', alpha=0.3)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title("Curves on the Sphere")
ax1.legend()

# ---- Second plot: Stereographic projection ----
ax2 = fig.add_subplot(122)
ax2.plot(x1_proj, y1_proj, 'b', label='Curve 1 (Projected)')
ax2.plot(x2_proj, y2_proj, 'r', label='Curve 2 (Projected)')
ax2.scatter([0], [0], color='black', s=50, label='Intersection')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title("Stereographic Projection")
ax2.legend()
ax2.grid()

plt.show()

# Part B 
print("Part b")


theta = np.linspace(0, 2 * np.pi, 200)  # Parameter for circles

# Great circle 1: Through the poles (Meridian at phi = 0)
x1 = np.sin(theta)
y1 = np.zeros_like(theta)
z1 = np.cos(theta)

# Great circle 2: Tilted (45-degree tilt)
tilt_angle = np.pi / 4  # 45-degree tilt
x2 = np.cos(theta)
y2 = np.sin(theta) * np.cos(tilt_angle)
z2 = np.sin(theta) * np.sin(tilt_angle)

# Great circle 3: Equator (xy-plane)
x3 = np.cos(theta)
y3 = np.sin(theta)
z3 = np.zeros_like(theta)

# Apply stereographic projection
x1_proj, y1_proj = stereographic_projection(x1, y1, z1)
x2_proj, y2_proj = stereographic_projection(x2, y2, z2)
x3_proj, y3_proj = stereographic_projection(x3, y3, z3)

# Create figure
fig = plt.figure(figsize=(12, 6))

# ---- First plot: Great circles on the sphere ----
ax1 = fig.add_subplot(121, projection='3d')

ax1.plot(x1, y1, z1, 'r', label='Meridian (Through Poles)')
ax1.plot(x2, y2, z2, 'g', label='Tilted Great Circle')
ax1.plot(x3, y3, z3, 'b', label='Equator')

# Sphere surface
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones(np.size(u)), np.cos(v))
ax1.plot_surface(X, Y, Z, color='gray', alpha=0.3)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title("Great Circles on the Sphere")
ax1.legend()

# ---- Second plot: Stereographic projection ----
ax2 = fig.add_subplot(122)

ax2.plot(x1_proj, y1_proj, 'r', label='Meridian Projection')
ax2.plot(x2_proj, y2_proj, 'g', label='Tilted Projection')
ax2.plot(x3_proj, y3_proj, 'b', label='Equator Projection')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title("Stereographic Projection of Great Circles")
ax2.legend()
ax2.grid()

plt.show()

