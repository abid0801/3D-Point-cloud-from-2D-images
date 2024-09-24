import open3d as o3d

pcd = o3d.io.read_point_cloud("point_cloud.ply")

o3d.visualization.draw_geometries([pcd], window_name="Loaded Point Cloud", width=800, height=600)
