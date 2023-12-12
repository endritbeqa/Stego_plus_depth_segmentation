import PIL.Image
from torchvision import transforms as T
from PIL import Image
import torch
import  numpy as np
import open3d as o3d


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def get_transform(res, is_target=False):
    sizeTransform = T.Resize(res, Image.NEAREST)
    cropper = T.CenterCrop(res)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if is_target:
        return T.Compose([sizeTransform, cropper,ToTargetTensor()])
    else:
        return T.Compose([sizeTransform, cropper, T.ToTensor(), normalize])


def invert_ridgit_body_transform(matrix):
    rotation = matrix[:3,:3]
    translation = matrix[:3,3:]

    inv_rotation = np.invert(rotation)
    inv_translation = -translation

    return inv_rotation, inv_translation

def get_pointCloud_transform(points, cam0_to_world, camera_intrinsics):#TODO finish this, maybe use the kitti360scripts


    R = cam0_to_world[:3, :3]
    T = cam0_to_world[:3, 3]

    R = np.expand_dims(R, 0)
    T = np.reshape(T, [1, -1, 3])
    points = np.expand_dims(points, 0)

    points_cam0 = np.matmul(R.transpose(0,2,1), (points -T).transpose(0,2,1))

    points_projected = np.matmul(camera_intrinsics[:3,:3].reshape([1,3,3]),points_cam0)

    depth = points_projected[:,2,:]
    depth[depth==0] = -1e-6
    u= np.round(points_projected[:,0,:]/np.abs(depth)).astype(np.int)
    v= np.round(points_projected[:,1,:]/np.abs(depth)).astype(np.int)

    u = u[0]
    v = v[0]
    depth = depth[0]

    valid_u_indexes = np.where((u>=0) & (u<1408))
    valid_v_indexes = np.where((v>=0) & (v<376))

    valid_indexes = np.intersect1d(valid_v_indexes, valid_u_indexes)

    depth_image = np.zeros((376,1408))

    for i in valid_indexes:
        depth_image[v[i],u[i]] = depth[i]

    depth_image = np.where(depth_image<0,0,depth_image)

    return depth_image




























