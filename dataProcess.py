## method1
# import os
# import nibabel as nib
# import numpy as np
# from nilearn.image import resample_to_img, resample_img
#
# # 定义目标体积和体素大小
# target_shape = (160, 214, 176)
# target_affine = np.eye(4)  # 1mm isotropic voxels
#
# # 定义输入和输出目录
# input_dir = 'PVS/preprocessed_t1'
# output_dir = 'mydata'
#
# # 创建输出目录
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # 遍历输入目录中的所有文件
# for filename in os.listdir(input_dir):
#     if filename.endswith('.nii.gz'):
#         # 加载图像
#         img_path = os.path.join(input_dir, filename)
#         img = nib.load(img_path)
#
#         # 检查图像的分辨率是否接近1mm isotropic
#         voxel_size = np.diag(img.affine)[:3]
#         if not np.allclose(voxel_size, [1., 1., 1.], atol=0.1):
#             # 如果分辨率不是1mm isotropic，则进行重采样
#             img = resample_img(img, target_affine=target_affine, target_shape=target_shape, interpolation='continuous')
#         else:
#             # 如果分辨率接近1mm isotropic，则进行裁剪
#             data = img.get_fdata()
#             current_shape = data.shape
#             start = [(current_shape[i] - target_shape[i]) // 2 for i in range(3)]
#             end = [start[i] + target_shape[i] for i in range(3)]
#             data = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
#             img = nib.Nifti1Image(data, img.affine)
#
#         # 将体素值归一化到0到1之间
#         data = img.get_fdata()
#         data = (data - np.min(data)) / (np.max(data) - np.min(data))
#         img = nib.Nifti1Image(data, img.affine)
#
#         # 保存处理后的图像
#         output_path = os.path.join(output_dir, filename)
#         nib.save(img, output_path)
#         print(f"Processed and saved: {output_path}")

### method 2
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

# 定义目标体积大小
target_shape = (160, 214, 176)

# 定义输入和输出目录
input_mask_dir = 'PVS/masks'
input_preprocessed_t1_dir = 'PVS/preprocessed_t1'
output_mask_dir = 'mydata/masks'
output_preprocessed_t1_dir = 'mydata/preprocessed_t1'


# 创建输出目录
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_preprocessed_t1_dir, exist_ok=True)


def resample_image(image, target_shape, is_mask=False):
    # 计算缩放因子
    zoom_factors = (
        target_shape[0] / image.shape[0],
        target_shape[1] / image.shape[1],
        target_shape[2] / image.shape[2]
    )

    # 对于mask文件，使用最近邻插值（order=0）
    if is_mask:
        resampled_image = zoom(image, zoom_factors, order=0)
        # 确保mask是二值的
        resampled_image = (resampled_image > 0).astype(np.uint8)
    else:
        # 对于其他文件，使用线性插值（order=1）并归一化
        resampled_image = zoom(image, zoom_factors, order=1)
        resampled_image = (resampled_image - resampled_image.min()) / (resampled_image.max() - resampled_image.min())

    return resampled_image


def process_directory(input_dir, output_dir, is_mask=False):
    for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz'):
            # 读取NIfTI文件
            filepath = os.path.join(input_dir, filename)
            img = nib.load(filepath)
            data = img.get_fdata()

            # 重采样图像
            resampled_data = resample_image(data, target_shape, is_mask=is_mask)

            # 创建新的NIfTI图像
            resampled_img = nib.Nifti1Image(resampled_data, img.affine, img.header)

            # 保存重采样后的图像
            output_filepath = os.path.join(output_dir, filename)
            nib.save(resampled_img, output_filepath)

            print(f"Processed and saved: {output_filepath}")


# 处理masks目录（使用最近邻插值，不归一化）
process_directory(input_mask_dir, output_mask_dir, is_mask=True)

# 处理preprocessed_t1目录（使用线性插值，归一化）
process_directory(input_preprocessed_t1_dir, output_preprocessed_t1_dir, is_mask=False)