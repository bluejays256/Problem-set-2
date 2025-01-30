import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
### --------------- PART A -----------

#Spherical: (1, /theta, /phi)
#Cartesian: (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
#Cylindrical: (sin(theta), phi, cos(theta))



def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.acos(z / r) if r != 0 else 0
    phi = np.atan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian_to_cylindrical(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    phi = np.atan2(y, x)
    return rho, phi, z

def cylindrical_to_cartesian(rho, phi, z):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z

def spherical_to_cylindrical(r, theta, phi):
    rho = r * np.sin(theta)
    z = r * np.cos(theta)
    return rho, phi, z

def cylindrical_to_spherical(rho, phi, z):
    r = np.sqrt(rho**2 + z**2)
    theta = np.atan2(rho, z) if r != 0 else 0
    return r, theta, phi


#-------------------- Part b ----------------------#
print("Part B")


# Generate a unit sphere
phi = np.linspace(0, 2 * np.pi, 30)  # Azimuthal angle
theta = np.linspace(0, np.pi, 30)    # Polar angle
phi, theta = np.meshgrid(phi, theta)

# Convert to Cartesian coordinates

x, y, z = spherical_to_cartesian(1, theta, phi)

# Define basis locations (vertices of an inscribed cube)
basis_points = np.array([
    [1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1],
    [1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]
]) / np.sqrt(3)  # Normalize to unit sphere

# Function to compute tangent vectors
def compute_tangent_basis(normal):
    """Computes two tangent vectors orthogonal to the given normal vector."""
    # Choose an arbitrary vector not parallel to the normal
    arbitrary = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])

    # Use Gram-Schmidt to find first tangent vector
    t1 = arbitrary - np.dot(arbitrary, normal) * normal
    t1 /= np.linalg.norm(t1)  # Normalize

    # Second tangent vector using cross product
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2)  # Normalize

    return normal, t1, t2  # Normal is already normalized

# Create figure and 3D axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
ax.plot_surface(x, y, z, color='c', alpha=0.3, edgecolor='k')

# Plot the basis vectors at each vertex
for point in basis_points:
    N, T1, T2 = compute_tangent_basis(point)  # Compute normal and tangent vectors

    ax.quiver(*point, *N, color='r', length=0.3, linewidth=2)  # Normal (outward)
    ax.quiver(*point, *T1, color='g', length=0.3, linewidth=2)  # Tangent 1
    ax.quiver(*point, *T2, color='b', length=0.3, linewidth=2)  # Tangent 2

# Labels and view adjustments
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Unit Sphere with Tangent Plane Bases')

# Set equal aspect ratio
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1,1,1])

plt.show()

# --------------Part C ---------------------#
print("Part c")
print("It is possible.")

# Generate unit sphere coordinates
phi_vals = np.linspace(0, 2 * np.pi, 30)  # Azimuthal angle
theta_vals = np.linspace(0, np.pi, 30)    # Polar angle
phi, theta = np.meshgrid(phi_vals, theta_vals)

# Convert to Cartesian coordinates for plotting the sphere
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Select a few points for basis visualization (e.g., 5x5 grid on sphere)
theta_samples = np.linspace(0.2, np.pi - 0.2, 5)  # Avoid poles
phi_samples = np.linspace(0, 2 * np.pi, 8)

basis_points = [(t, p) for t in theta_samples for p in phi_samples]

# Create figure and 3D axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
ax.plot_surface(x, y, z, color='c', alpha=0.3, edgecolor='k')

# Plot basis vectors at selected points
for theta, phi in basis_points:
    # Convert to Cartesian coordinates (point on the sphere)
    x0 = np.sin(theta) * np.cos(phi)
    y0 = np.sin(theta) * np.sin(phi)
    z0 = np.cos(theta)

    # Compute spherical basis vectors
    e_r = np.array([x0, y0, z0])  # Radial vector
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])  # Tangent along theta
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])  # Tangent along phi

    # Plot basis vectors
    ax.quiver(x0, y0, z0, *e_r, color='r', length=0.2, linewidth=2)    # Radial (Red)
    ax.quiver(x0, y0, z0, *e_theta, color='g', length=0.2, linewidth=2) # Theta (Green)
    ax.quiver(x0, y0, z0, *e_phi, color='b', length=0.2, linewidth=2)   # Phi (Blue)

# Labels and view adjustments
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Unit Sphere with Spherical Basis Vectors')

# Set equal aspect ratio
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1,1,1])

plt.show()

# --------------- Part d --------------#
print("Part d")
def local_coordinate_system(f, x_range, y_range, num_points=10):
    """
    Plots the surface z = f(x, y) and overlays the local coordinate system (normal & tangent vectors).
    
    Parameters:
        f: function - The surface function f(x, y).
        x_range: tuple - The range (min, max) for x.
        y_range: tuple - The range (min, max) for y.
        num_points: int - Number of sample points for vector field visualization.
    """
    # Generate grid for the surface
    x = np.linspace(*x_range, 50)
    y = np.linspace(*y_range, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Compute gradients (partial derivatives)
    fx, fy = np.gradient(Z, x, axis=1), np.gradient(Z, y, axis=0)

    # Sample points for vector visualization
    xs = np.linspace(*x_range, num_points)
    ys = np.linspace(*y_range, num_points)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = f(Xs, Ys)

    # Compute normal and tangent vectors at sample points
    fx_s, fy_s = np.gradient(Zs, xs, axis=1), np.gradient(Zs, ys, axis=0)
    normals = np.dstack((-fx_s, -fy_s, np.ones_like(fx_s)))  # Normal vectors
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)  # Normalize

    # Tangent vectors along x and y
    tangent_x = np.dstack((np.ones_like(fx_s), np.zeros_like(fx_s), fx_s))
    tangent_y = np.dstack((np.zeros_like(fy_s), np.ones_like(fy_s), fy_s))
    tangent_x /= np.linalg.norm(tangent_x, axis=2, keepdims=True)
    tangent_y /= np.linalg.norm(tangent_y, axis=2, keepdims=True)

    # Plot surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='c', alpha=0.5, edgecolor='k')

    # Plot normal and tangent vectors
    for i in range(num_points):
        for j in range(num_points):
            x0, y0, z0 = Xs[i, j], Ys[i, j], Zs[i, j]
            nx, ny, nz = normals[i, j]
            tx, ty, tz = tangent_x[i, j]
            ux, uy, uz = tangent_y[i, j]

            ax.quiver(x0, y0, z0, nx, ny, nz, color='r', length=0.2, linewidth=2)  # Normal vector
            ax.quiver(x0, y0, z0, tx, ty, tz, color='g', length=0.2, linewidth=2)  # Tangent x
            ax.quiver(x0, y0, z0, ux, uy, uz, color='b', length=0.2, linewidth=2)  # Tangent y

    # Labels and formatting
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Surface with Local Coordinate System")

    plt.show()

# Example usage
f = lambda x, y: np.sin(x) * np.cos(y)  # Example surface
local_coordinate_system(f, x_range=(-2, 2), y_range=(-2, 2), num_points=8)

# ------------------------- Part e --------------------------- #


def spherical_to_cartesian(r, theta, phi):
    """Converts spherical coordinates to Cartesian (x, y, z)."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def parallel_transport_on_sphere():
    """Performs and visualizes the parallel transport of a vector on the unit sphere along a meridian."""
    # Define transport path along the meridian (constant phi=0)
    theta_vals = np.linspace(np.pi/5, np.pi/2, 10)  # Interpolated points
    phi = 0  # Constant meridian
    r = 1  # Unit sphere

    # Compute positions on the sphere
    positions = np.array([spherical_to_cartesian(r, theta, phi) for theta in theta_vals])

    # Initial basis vectors at theta = pi/5
    theta0 = np.pi/5
    e_theta0 = np.array([np.cos(theta0), 0, -np.sin(theta0)])  # Tangent in theta direction
    e_phi0 = np.array([0, 1, 0])  # Tangent in phi direction

    # Transported vector perpendicular to motion 
    v_perp = e_phi0  # Stays aligned with e_phi

    # Transported vector along motion
    transported_vectors_along_motion = []
    transported_vectors_perpendicular = []

    for theta in theta_vals:
        # Compute the new tangent basis at theta
        e_theta = np.array([np.cos(theta), 0, -np.sin(theta)])  # Updated theta direction
        e_phi = np.array([0, 1, 0])  # Still same

        # Parallel transport: Keep the vector components constant in the moving basis
        v_along = e_theta  # Transported vector in direction of motion
        v_perp = e_phi  # Transported perpendicular vector

        transported_vectors_along_motion.append(v_along)
        transported_vectors_perpendicular.append(v_perp)

    # Convert vectors to Cartesian coordinates
    transported_vectors_along_motion_cartesian = np.array([
        spherical_to_cartesian(0, theta, phi) + v * 0.2
        for theta, v in zip(theta_vals, transported_vectors_along_motion)
    ])
    transported_vectors_perpendicular_cartesian = np.array([
        spherical_to_cartesian(0, theta, phi) + v * 0.2
        for theta, v in zip(theta_vals, transported_vectors_perpendicular)
    ])

    # Plot sphere and transported vectors
    fig = plt.figure(figsize=(12, 5))

    # Plot 1: Transported vector perpendicular to motion
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Parallel Transport (Perpendicular Vector)")
    
    # Plot 2: Transported vector along motion
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Parallel Transport (Along Motion)")

    # Sphere mesh
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot sphere on both subplots
    for ax in [ax1, ax2]:
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='c', alpha=0.3, edgecolor='k')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=2, label="Parallel Transport Path")

    # Plot transported perpendicular vectors (left plot)
    for i in range(len(theta_vals)):
        ax1.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                   transported_vectors_perpendicular_cartesian[i, 0], transported_vectors_perpendicular_cartesian[i, 1], transported_vectors_perpendicular_cartesian[i, 2],
                   color='b', length=1, linewidth=2)

    # Plot transported vectors along motion (right plot)
    for i in range(len(theta_vals)):
        ax2.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                   transported_vectors_along_motion_cartesian[i, 0], transported_vectors_along_motion_cartesian[i, 1], transported_vectors_along_motion_cartesian[i, 2],
                   color='g', length=4, linewidth=2)

    # Labels
    for ax in [ax1, ax2]:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

    plt.show()

# Run the transport visualization
parallel_transport_on_sphere()

#Part f) 




print("-=------------------------------")



def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def rotate_vector(v, angle, axis):
    """Rotate a vector by a given angle around a specified axis."""
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rotation matrix using Rodrigues' rotation formula
    rotation_matrix = np.array([
        [cos_angle + axis[0]**2 * (1 - cos_angle), axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle, axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle],
        [axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle, cos_angle + axis[1]**2 * (1 - cos_angle), axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle],
        [axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle, axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle, cos_angle + axis[2]**2 * (1 - cos_angle)]
    ])
    
    return np.dot(rotation_matrix, v)

def parallel_transport_phi(theta):
    """Parallel transport along a full 2π loop in phi at constant theta and visualize the process in separate plots."""
    theta = theta  # Fixed latitude (45 degrees)
    phi_vals = np.linspace(0, 2 * np.pi, 30)  # Full circle in phi
    r = 1  # Unit sphere

    # Initial tangent basis vectors at phi = 0
    e_theta0 = np.array([np.cos(theta), 0, -np.sin(theta)])  # Along θ
    e_phi0 = np.array([-np.sin(0), np.cos(0), 0])  # Along φ (circumferential)

    # Initial vectors
    v1_0 = e_theta0  # Aligned with θ
    v2_0 = e_phi0  # Aligned with φ
    v3_0 = (e_theta0 + e_phi0) / np.linalg.norm(e_theta0 + e_phi0)  # Normalized

    # Transported vectors
    transported_v1, transported_v2, transported_v3 = [], [], []
    vectors_at_phi = []

    for phi in phi_vals:
        # Update the basis vectors as phi changes
        e_phi = np.array([-np.sin(phi), np.cos(phi), 0])  # Circumferential direction
        e_theta = e_theta0  # Theta direction is fixed (not rotating)

        # The idea is that the vector rotates in the phi-direction, considering holonomy
        v1 = e_theta  # Transported v1 remains in the e_theta direction
        v2 = e_phi  # Transported v2 remains in the e_phi direction
        v3 = (e_theta + e_phi) / np.linalg.norm(e_theta + e_phi)  # Normalize the transported v3

        # Apply rotation due to holonomy for the vectors
        if phi > 0:
            # Rotation axis for the vectors is along the z-axis (vertical axis of the sphere)
            rotation_axis = np.array([0, 0, 1])
            angle = -np.pi * phi / (2 * np.pi)  # Angle is proportional to phi
            v1 = rotate_vector(v1, angle, rotation_axis)
            v2 = rotate_vector(v2, angle, rotation_axis)
            v3 = rotate_vector(v3, angle, rotation_axis)

        transported_v1.append(v1)
        transported_v2.append(v2)
        transported_v3.append(v3)
        vectors_at_phi.append(spherical_to_cartesian(r, theta, phi))

    # Create the plots
    fig = plt.figure(figsize=(15, 10))

    # Plot for vector 1 (theta-aligned)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title("Parallel Transport of v1 (θ-aligned)")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot the unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, color='b', alpha=0.1, rstride=5, cstride=5)

    # Plot the initial vector and transported vectors
    ax1.quiver(0, 0, 0, v1_0[0], v1_0[1], v1_0[2], color='r', label='Initial v1')
    for v1, pos in zip(transported_v1, vectors_at_phi):
        ax1.quiver(pos[0], pos[1], pos[2], v1[0], v1[1], v1[2], color='r', alpha=0.5)
    ax1.quiver(vectors_at_phi[-1][0], vectors_at_phi[-1][1], vectors_at_phi[-1][2], 
               transported_v1[-1][0], transported_v1[-1][1], transported_v1[-1][2], color='r', linewidth=2)

    # Plot for vector 2 (phi-aligned)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title("Parallel Transport of v2 (φ-aligned)")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Plot the unit sphere
    ax2.plot_surface(x, y, z, color='b', alpha=0.1, rstride=5, cstride=5)

    # Plot the initial vector and transported vectors
    ax2.quiver(0, 0, 0, v2_0[0], v2_0[1], v2_0[2], color='g', label='Initial v2')
    for v2, pos in zip(transported_v2, vectors_at_phi):
        ax2.quiver(pos[0], pos[1], pos[2], v2[0], v2[1], v2[2], color='g', alpha=0.5)
    ax2.quiver(vectors_at_phi[-1][0], vectors_at_phi[-1][1], vectors_at_phi[-1][2], 
               transported_v2[-1][0], transported_v2[-1][1], transported_v2[-1][2], color='g', linewidth=2)

    # Plot for vector 3 (mixed)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title("Parallel Transport of v3 (mixed)")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Plot the unit sphere
    ax3.plot_surface(x, y, z, color='b', alpha=0.1, rstride=5, cstride=5)

    # Plot the initial vector and transported vectors
    ax3.quiver(0, 0, 0, v3_0[0], v3_0[1], v3_0[2], color='b', label='Initial v3')
    for v3, pos in zip(transported_v3, vectors_at_phi):
        ax3.quiver(pos[0], pos[1], pos[2], v3[0], v3[1], v3[2], color='b', alpha=0.5)
    ax3.quiver(vectors_at_phi[-1][0], vectors_at_phi[-1][1], vectors_at_phi[-1][2], 
               transported_v3[-1][0], transported_v3[-1][1], transported_v3[-1][2], color='b', linewidth=2)

    # Show the plot
    plt.tight_layout()
    plt.show()

    dot_product_tuple = np.array([theta, np.dot(v3_0, transported_v3[-1])])
    print(dot_product_tuple)
    return dot_product_tuple

   

    
# Run the function to visualize
theta_ranges = np.linspace(0, np.pi, 8)
holonomy_compare = []
for i in range (0, len(theta_ranges) - 1):
    a = parallel_transport_phi(theta_ranges[i])
    holonomy_compare.append(a)
print(holonomy_compare)

theta_data = x = [point[0] for point in holonomy_compare]

dot_product_data = [point[1] for point in holonomy_compare]

plt.scatter(theta_data, dot_product_data)
plt.xlabel("\theta_0")
plt.ylabel("Dot product result")
plt.legend()
plt.show()

