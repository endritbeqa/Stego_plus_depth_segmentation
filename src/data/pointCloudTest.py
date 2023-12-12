import numpy as np
import open3d as o3d
import os


if __name__=="__main__":

    filePath = "/home/endrit/geo-scene/data/KITTI-360/data_3d_semantics/train/2013_05_28_drive_0000_sync/static/0000000002_0000000385.ply"
    pc =o3d.io.read_point_cloud(filePath)
    print(pc)
    pcNP=np.asarray(pc.points)
    print(pcNP)
    o3d.visualization.draw_geometries([pc])
    print("Breakpoint")




