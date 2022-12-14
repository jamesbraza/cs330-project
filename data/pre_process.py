import os.path
from argparse import ArgumentParser

import cv2
import numpy as np
from sklearn import preprocessing


def get_file_pair(label_path):
    folder_names = os.listdir(label_path)
    folder_path = [
        os.path.join(label_path, f) for f in folder_names if not f.startswith(".")
    ]

    x_path_list = []
    y_list = []
    for folder in folder_path:
        pic_path = [os.path.join(folder, f) for f in os.listdir(folder)]
        pic_path_1000 = pic_path[0:1000]
        for j in pic_path_1000:
            y = str(j).split("/")[-2]
            y_list.append(y)
            x_path_list.append(j)

    y_np = np.array(y_list)
    # ====label encode
    le = preprocessing.LabelEncoder()
    le.fit(y_np)
    y_encode = le.transform(y_np)

    return x_path_list, y_encode


def pre_process(file_names):
    img1 = cv2.resize(
        cv2.imread(file_names, 1), dsize=(28, 28), interpolation=cv2.INTER_AREA
    )
    return np.array(img1[..., ::-1] / 255.0, dtype="float32")


def convert_folder_numpy(args):
    label_path = args.label_path
    output_dir = args.output_dir

    x_path, y_label = get_file_pair(label_path)

    img_list = []
    for image_path in x_path:
        img_np = pre_process(image_path)
        img_list.append(img_np)

    img_np = np.array(img_list)
    print(">>>>your x shape", img_np.shape)
    print(">>>>>your y shape", y_label.shape)
    np.save(os.path.join(output_dir, "x4"), img_np)
    np.save(os.path.join(output_dir, "y4"), y_label)


def main():
    parser = ArgumentParser(description="Train parameters ")
    parser.add_argument(
        "--label_path",
        type=str,
        default="/Users/annaning/Desktop/cs330/project/data/find_tune_val",
        help="input",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/annaning/Desktop/cs330/project/data/find_tune_val",
        help="output_dir",
    )
    convert_folder_numpy(args=parser.parse_args())


if __name__ == "__main__":
    main()
