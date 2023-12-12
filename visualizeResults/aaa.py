import numpy as np
import open3d as o3d
import cv2



def project_disparity_to_3d(depth_map):
    focal_length_x = 2268.36
    focal_length_y = 2225.5405988775956
    cx = 1048.64
    cy = 519.277

    point_cloud = []

    height, width = depth_map.shape

    # Generate a grid of pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))


    valid_indices = np.where(depth_map != 0.5)
    depth = depth_map[valid_indices]*1000/0.2645833333
    points_x = ((grid_x[valid_indices] - cx)) * (depth / focal_length_x)
    points_y = ((grid_y[valid_indices] - cy)) * (depth / focal_length_y)
    points_z = depth


    # Stack the coordinates into a point cloud
    point_cloud = np.stack((points_x, points_y, points_z), axis=-1)

    return point_cloud





if __name__ == "__main__":



    disparity = cv2.imread("/data/cityscapes/disparity/val/frankfurt/frankfurt_000000_000576_disparity.png", cv2.IMREAD_UNCHANGED).astype(np.float64)
    disparity = np.where(disparity > 10, (disparity - 1) / 256, 0)
    depth = np.where(disparity > 0, (0.222126 * 2268.36) / disparity, 0)

    projected_pointcloudzzz = project_disparity_to_3d(depth)
    #normal_pointCloudzzz = create_all_point_clouds(depth)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(projected_pointcloudzzz)
    #pcd.points = o3d.utility.Vector3dVector(normal_pointCloudzzz)
    o3d.visualization.draw_geometries([pcd])
