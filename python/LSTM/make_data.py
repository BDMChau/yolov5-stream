import cv2
import pandas as pd
from ultralytics import YOLO
import torch
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# config YOLO
modelPersonPose = YOLO(parent_dir + "/weights/yolov8l-pose.pt").to(device)


def make_data_from_video(path, file_name):
    data_to_write = []

    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            print("Error: Unable to read frame.")
            break

        modelPersonPoseResults = modelPersonPose(
            img,
            stream=True,
            conf=0.2
        )

        for r in modelPersonPoseResults:
            img = r.plot(kpt_line=True, kpt_radius=5)
 
            # POSE
            keypoints_xyn = r.keypoints.xyn[0]  # just get the first
            result_timestep = []
            print("r.keypoints",r.keypoints)
            for i, keypoint_tensor in enumerate(keypoints_xyn):
                print("keypoint_tensor", keypoint_tensor)
                x, y = (
                    keypoint_tensor[0].item(),
                    keypoint_tensor[1].item(),
                )

                result_timestep.append(x)
                result_timestep.append(y)

            data_to_write.append(result_timestep)

        cv2.imshow("image", img)
        cv2.waitKey(1)

    # save to csv file
    print("Writing file...")
    df = pd.DataFrame(data_to_write)
    df.to_csv("files/" + file_name + ".csv")

    cap.release()
    cv2.destroyAllWindows()


def make_data_from_imgs(folder_path, file_name):
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.lower().endswith((".jpg", ".jpeg", ".png"))]

    data_to_write = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        img = cv2.imread(image_path)

        if img is not None:
            print(f"image OK: {image_path}")
            modelPersonPoseResults = modelPersonPose(
                img,
                stream=True,
                conf=0.2
            )

            for r in modelPersonPoseResults:
                img = r.plot(kpt_line=True, kpt_radius=5)
    
                # POSE
                keypoints_xyn = r.keypoints.xyn[0]  # just get the first
                result_timestep = []
                for i, keypoint_tensor in enumerate(keypoints_xyn):
                    x, y = (
                        keypoint_tensor[0].item(),
                        keypoint_tensor[1].item(),
                    )

                    result_timestep.append(x)
                    result_timestep.append(y)

                data_to_write.append(result_timestep)
                
                cv2.imshow("image", img)
                cv2.waitKey(1)
        else:
            print(f"Failed to load the image: {image_path}")
            
     # save to csv file
    print("Writing file...")
    df = pd.DataFrame(data_to_write)
    df.to_csv("files/" + file_name + ".csv")
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    file_name = "picking"
    
    make_data_from_video(parent_dir + "/data/picking.mp4", file_name)
    # make_data_from_imgs(parent_dir + "/data/imgs", file_name)
