# import os
# import numpy as np
# from numpy.lib.format import open_memmap
# import random

# sets = {
#     'train', 'val'
# }

# # 'ntu/xview', 'ntu/xsub'
# datasets_path = '/data/xcl_data/nturgb+d_rgb/nturgb+d_60_rgb/nturgb+d_frames/'

# from tqdm import tqdm

# for dataset in os.listdir(datasets_path):    # dataset is data_dir just like S009C003P007R002A060
#     for data in dataset:         # there are many .jpg in data
#         print(dataset, data)
#         data = np.load('/data/xcl_data/{}/{}_data_joint.npy'.format(dataset, set1))
#         N, C, T, V, M = data.shape
#         T1 = T // 2                     # T1 = 150
#         # frames_per_group = T // T1    # Frames per group=5



#         # reverse = open_memmap(
#         #     '../data/{}/{}_data_joint_cut_150.npy'.format(dataset, set1),
#         #     dtype='float32',
#         #     mode='w+',
#         #     shape=(N, 3, T1, V, M))
#         # reverse[:, :, :T1, :, :] = data[:, :, ::2, :, :]


#         reverse = open_memmap(
#             '/data/xcl_data/{}/{}_data_joint_FR.npy'.format(dataset, set1),  # 修改保存的文件名
#             dtype='float32',
#             mode='w+',
#             shape=(N, 3, T, V, M))
#         for i in range(T1):
#             frame_indices = random.sample(range(i * 2, (i + 1) * 2), 1)  # 从每五帧中随机选择一帧
#             reverse[:, :, i, :, :] = data[:, :, frame_indices[0], :, :]
#         reverse[:, :, (i+1):, :, :] = reverse[:, :, T:(T1-1):-1, :, :]



# import os
# from PIL import Image
# import shutil

# datasets_path = '/data/xcl_data/nturgb+d_rgb/nturgb+d_60_rgb/nturgb+d_frames/'
# output_path = '/data/xcl_data/nturgb+d_rgb/nturgb+d_60_rgb/NTU60_RGB_MIS/'

# folders = [folder for folder in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, folder))]

# for folder in folders:
#     folder_path = os.path.join(datasets_path, folder)
#     output_folder_path = os.path.join(output_path, folder)  # 新文件夹路径
    
#     os.makedirs(output_folder_path, exist_ok=True)  # 创建新文件夹

#     files_in_folder = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

#     # 获取所有以 .jpg 结尾的文件并按照序号递增顺序排序
#     jpg_files = sorted([f for f in files_in_folder if f.lower().endswith('.jpg')], key=lambda x: int(os.path.splitext(x)[0]))

#     # 选择中间的一个文件
#     if jpg_files:
#         middle_index = len(jpg_files) // 2
#         start_index = len(middle_index) // 2
#         middle_file = jpg_files[middle_index]
#         source_path = os.path.join(folder_path, middle_file)
#         destination_path = os.path.join(output_folder_path, middle_file)
        
#         # 复制中间文件到新文件夹
#         shutil.copyfile(source_path, destination_path)
#         print(f"Saved middle image from {folder}")

# print("All middle images saved successfully.")




# import os
# from PIL import Image
# import shutil

# datasets_path = '/data/xcl_data/nturgb+d_rgb/nturgb+d_60_rgb/nturgb+d_frames/'
# output_path = '/data/xcl_data/nturgb+d_rgb/nturgb+d_60_rgb/NTU60_RGB_MIS/'

# folders = [folder for folder in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, folder))]

# for folder in folders:
#     folder_path = os.path.join(datasets_path, folder)
#     files_in_folder = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
#     # 获取所有以 .jpg 结尾的文件并按照序号递增顺序排序
#     jpg_files = sorted([f for f in files_in_folder if f.lower().endswith('.jpg')], key=lambda x: int(os.path.splitext(x)[0]))

#     # 按照文件数量将文件分成三部分
#     num_files = len(jpg_files)
#     part_size = num_files // 3
    
#     for i in range(3):
#         start_idx = i * part_size
#         end_idx = (i + 1) * part_size if i < 2 else num_files  # 处理最后一个部分的边界情况

#         # 获取当前部分的中间图片
#         middle_index = (start_idx + end_idx) // 2
#         middle_file = jpg_files[middle_index]
        
#         # 创建新文件夹以保存新图片
#         new_folder_path = os.path.join(output_path, folder)
#         os.makedirs(new_folder_path, exist_ok=True)
        
#         # 复制并保存中间的一个图片到新文件夹中
#         source_file_path = os.path.join(folder_path, middle_file)
#         destination_file_path = os.path.join(new_folder_path, middle_file)
#         shutil.copyfile(source_file_path, destination_file_path)
#         print(f"Saved middle image from part {i + 1} for folder {folder} as {destination_file_path}")


# from PIL import Image
# import os

# output_path = '/data/xcl_data/nturgb+d_rgb/nturgb+d_60_rgb/NTU60_RGB_MIS/'

# folders = [folder for folder in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, folder))]

# for folder in folders:
#     folder_path = os.path.join(output_path, folder)
#     files_in_folder = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

#     # 获取所有以 .jpg 结尾的文件并按照序号递增顺序排序
#     jpg_files = sorted([f for f in files_in_folder if f.lower().endswith('.jpg')], key=lambda x: int(os.path.splitext(x)[0]))

#     for file in jpg_files:
#         file_path = os.path.join(folder_path, file)

#         # 打开图片
#         img = Image.open(file_path)

#         # 定义裁剪尺寸
#         width, height = img.size
#         left = width // 3
#         top = height // 6
#         right = width - left
#         bottom = height - top

#         # 进行裁剪
#         cropped_img = img.crop((left, top, right, bottom))

#         # 重新保存图片
#         new_file_path = os.path.join(folder_path, file)
#         cropped_img.save(new_file_path)
#         print(f"Processed and saved {new_file_path}")


from PIL import Image
import os
import shutil

def combine_images_in_folder(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in subfolders:
        image_files = [file.path for file in os.scandir(subfolder) if file.name.endswith('.jpg')]
        
        if len(image_files) != 3:
            print(f"文件夹 {subfolder} 中图片数量不是3张, 无法拼接。")
            continue
        
        images = [Image.open(img) for img in image_files]
        images.sort(key=lambda x: min(x.size), reverse=True)
        
        width = sum(img.size[0] for img in images)
        height = max(img.size[1] for img in images)
        combined_image = Image.new('RGB', (width, height))
        
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        # 保存为与子文件夹同名的 .png 文件
        combined_image.save(os.path.join(folder_path, f"{os.path.basename(subfolder)}.png"))
        print(f"成功拼接并保存文件夹 {subfolder} 中的图片为 .png 文件。")
        
        # 删除原始子文件夹
        shutil.rmtree(subfolder)
        print(f"已删除原始子文件夹 {subfolder}。")

folder_to_combine = '/data/xcl_data/nturgb+d_rgb/nturgb+d_60_rgb/NTU60_RGB_MIS'
combine_images_in_folder(folder_to_combine)

