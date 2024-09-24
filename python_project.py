import torch
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.to(device)
model.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

img_path = 'apple.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

target_width, target_height = img_rgb.shape[1] // 4, img_rgb.shape[0] // 4
img_rgb_small = cv2.resize(img_rgb, (target_width, target_height))

input_batch = transform(img_rgb_small).to(device)
with torch.no_grad():
    depth_prediction = model(input_batch)

depth_map = depth_prediction.squeeze().cpu().numpy()
depth_map_small = cv2.resize(depth_map, (target_width, target_height))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_rgb_small)

plt.subplot(1, 2, 2)
plt.title("Depth Map")
plt.imshow(depth_map_small, cmap='plasma')
plt.colorbar(label='Depth')
plt.savefig("image.jpg")


def generate_point_cloud(depth_map, img, scale=0.05):
    h, w = depth_map.shape
    point_cloud = []
    rgb_colors = []
    
    for y in range(h):
        for x in range(w):
            z = depth_map[y, x] * scale
            X = (x - w // 2) * z / 650
            Y = -(y - h // 2) * z / 650
            point_cloud.append([X, Y, z])
            rgb_colors.append(img[y, x] / 255.0)
    
    return np.array(point_cloud, dtype=np.float64), np.array(rgb_colors, dtype=np.float64)


point_cloud, rgb_colors = generate_point_cloud(depth_map_small, img_rgb_small)

pcd = o3d.geometry.PointCloud()
print("Assigning points to point cloud")
pcd.points = o3d.utility.Vector3dVector(point_cloud)
print("Points assigned")

print("Assigning colors to point cloud")
pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
print("Colors assigned")

o3d.io.write_point_cloud("point_cloud.ply", pcd)
print("Point cloud saved as 'point_cloud.ply'.")
