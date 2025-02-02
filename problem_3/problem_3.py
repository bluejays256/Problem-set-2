import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial import Delaunay
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D

point_cloud = np.loadtxt("/root/Desktop/host/mesh.dat", skiprows = 1)

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def graham_scan(point_cloud):
    #sorted function uses mergesort algorithm.
    point_cloud = [tuple(p) for p in point_cloud] #converts a numpy array to a list
    point_cloud = sorted(point_cloud)

    lower = [] #initialize hull stack
    for p in point_cloud:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []

    for p in reversed(point_cloud):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower + upper

tri = Delaunay(point_cloud)
plt.triplot(point_cloud[:, 0], point_cloud[:, 1], tri.simplices, color='blue')
graham_scan_result = graham_scan(point_cloud)
plt.scatter(*zip(*point_cloud))
plt.plot(*zip(*graham_scan_result), linestyle = '-', color = 'blue')
plt.show()

# Step 3: Apply the lifting map z = x^2 + y^2
def lifting_map(p):
    return p[0]**2 + p[1]**2

# Step 4: Functions to calculate areas
def triangle_area_2d(a, b, c):
    return 0.5 * np.abs(np.cross(b - a, c - a))

def triangle_area_3d(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

# Step 5: Calculate area ratios and prepare triangles for plotting
triangles = []
area_ratios = []

for simplex in tri.simplices:
    # Original 2D point_cloud
    a, b, c = point_cloud[simplex]

    # Lifted point_cloud (in 3D)
    A = np.array([a[0], a[1], lifting_map(a)])
    B = np.array([b[0], b[1], lifting_map(b)])
    C = np.array([c[0], c[1], lifting_map(c)])

    # Areas
    area_2d = triangle_area_2d(a, b, c)
    area_3d = triangle_area_3d(A, B, C)

    # Area ratio
    area_ratio = area_2d / area_3d

    # Store triangles and their area ratios
    triangles.append([a, b, c])
    area_ratios.append(area_ratio)

# Step 6: Plotting the heatmap with colored triangles
fig, ax = plt.subplots(figsize=(8, 6))

# Create a PolyCollection to color each triangle
collection = PolyCollection(triangles, array=np.array(area_ratios),
                            cmap='viridis', edgecolors='k', linewidths=0.5)

ax.add_collection(collection)

# Add colorbar
cbar = plt.colorbar(collection, ax=ax)
cbar.set_label('Area Ratio (2D / 3D)')

# Plot point_cloud for reference
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], color='red', s=10)

# Final plot adjustments
ax.set_title('Change in Area After Lifting (Heatmap)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')

plt.show()

# Part C #



#Part D #


# Step 3: Function to calculate the surface normal of a triangle
def calculate_normal(A, B, C):
    AB = B - A
    AC = C - A
    normal = np.cross(AB, AC)
    return normal / np.linalg.norm(normal)  # Normalize the vector

# Step 4: Calculate lifted point_cloud and normals
lifted_point_cloud = np.column_stack((point_cloud, lifting_map(point_cloud.T)))
normals = []
centroids = []

for simplex in tri.simplices:
    # Triangle vertices (lifted in 3D)
    A = lifted_point_cloud[simplex[0]]
    B = lifted_point_cloud[simplex[1]]
    C = lifted_point_cloud[simplex[2]]

    # Centroid for plotting normals
    centroid = (A + B + C) / 3
    centroids.append(centroid)

    # Surface normal
    normal = calculate_normal(A, B, C)
    normals.append(normal)

centroids = np.array(centroids)
normals = np.array(normals)

# Step 5: Plotting the lifted mesh with surface normals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the lifted mesh
ax.plot_trisurf(lifted_point_cloud[:, 0], lifted_point_cloud[:, 1], lifted_point_cloud[:, 2], 
                triangles=tri.simplices, cmap='viridis', alpha=0.6, edgecolor='gray')

# Plot the surface normals
ax.quiver(centroids[:, 0], centroids[:, 1], centroids[:, 2],
          normals[:, 0], normals[:, 1], normals[:, 2],
          length=0.5, color='red', linewidth=1)

# Axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('Surface Normals of the Lifted Mesh')
plt.show()

#Part E #
def triangle_area(A, B, C):
    return 0.5 * np.linalg.norm(np.cross(B - A, C - A))
lifted_point_cloud = np.column_stack((point_cloud, lifting_map(point_cloud.T)))

# Step 5: Compute face normals and areas
face_normals = []
face_areas = []

for simplex in tri.simplices:
    A, B, C = lifted_point_cloud[simplex]
    normal = calculate_normal(A, B, C)
    area = triangle_area(A, B, C)
    face_normals.append(normal)
    face_areas.append(area)

face_normals = np.array(face_normals)
vertex_normals = np.zeros_like(lifted_point_cloud)

for i, simplex in enumerate(tri.simplices):
    for vertex in simplex:
        vertex_normals[vertex] += face_normals[i] * face_areas[i]  # Area-weighted sum

# Normalize the vertex normals
vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True)

# Step 7: Plotting the lifted mesh with vertex normals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the lifted mesh
ax.plot_trisurf(lifted_point_cloud[:, 0], lifted_point_cloud[:, 1], lifted_point_cloud[:, 2],
                triangles=tri.simplices, cmap='viridis', alpha=0.6, edgecolor='gray')

# Plot the vertex normals
ax.quiver(lifted_point_cloud[:, 0], lifted_point_cloud[:, 1], lifted_point_cloud[:, 2],
          vertex_normals[:, 0], vertex_normals[:, 1], vertex_normals[:, 2],
          length=0.5, color='blue', linewidth=1)

# Axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('Vertex Normals of the Lifted Mesh')
plt.show()

