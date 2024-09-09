import os
import cv2

framestart = 1
frameend = 1
frame_count = 0
file_count = 0

sss = 1  # setup_id
ccc = 1  # camera_id
ppp = 1  # subject_id
rrr = 1  # duplicate_id
aaa = 1  # action_id

for setup_id in range(1, 18):  # 1:20 Different height and distance
    if setup_id < sss:
        continue
    for camera_id in range(1, 4):  # 1:3 camera views
        if setup_id < sss + 1 and camera_id < ccc:
            continue
        for subject_id in range(1, 41):  # 1:40 distinct subjects aged between 10 to 35
            if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp:
                continue
            for duplicate_id in range(1, 3):  # 1:2 Performance action twice, one to left camera, one to right camera
                if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp + 1 and duplicate_id < rrr:
                    continue
                for action_id in range(1, 61):  # 1:60 Action class
                    if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp + 1 and duplicate_id < rrr + 1 and action_id < aaa:
                        continue

                    skeleton_file_name = f"S{setup_id:03d}C{camera_id:03d}P{subject_id:03d}R{duplicate_id:03d}A{action_id:03d}"
                    file_name_to_save = os.path.join("data", "xcl_data", "nturgb+d_rgb", "nturgb+d_60_rgb", "nturgb+d_rgb", f"{skeleton_file_name}_rgb.avi")

                    if os.path.isfile(file_name_to_save):
                        file_count += 1
                        action_folder = os.path.join("data", "xcl_data", "nturgb+d_rgb", "nturgb+d_60_rgb", "nturgb+d_frames", skeleton_file_name)
                        os.makedirs(action_folder, exist_ok=True)

                        cap = cv2.VideoCapture(file_name_to_save)
                        frameNumber = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            image_name = os.path.join(action_folder, f"{frameNumber:04d}.jpg")
                            cv2.imwrite(image_name, frame)
                            frameNumber += 1

                        cap.release()
                    else:
                        # File does not exist
                        pass

                    skeleton_file_name = ''
                    if file_count % 500 == 0 and file_count != 0:
                        print(f"file_count: {file_count}")
