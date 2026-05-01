import cv2

from config import get_settings
from check_path_exits import check_path_exits
from verify_image import is_blurry, is_corrupted

settings = get_settings()
import os

"""""
This file is responsible fr
"""
from PIL import Image

import os
from PIL import Image

settings = get_settings()
"""""
This file is responsible fr
"""


def preprocessing(data_path: str):
    if check_path_exits(data_path) is False:
        print("Folder not found Error")
        return
    for label in os.listdir(data_path):
        raw_folder = os.path.join(data_path, label)
        processed_folder = os.path.join(settings.PROCESSED_DATA_PATH, label)

        os.makedirs(processed_folder, exist_ok=True)

        for img_name in os.listdir(raw_folder):
            img_path = os.path.join(raw_folder, img_name)

            try:
                img = cv2.imread(img_path)

                if img is None:
                    continue
                if is_corrupted(img_path):
                    print(f"🗑️ Deleting corrupted {img_name}")
                    os.remove(img_path)
                    continue
                if is_blurry(img_path):
                    print(f"🗑️ Deleting blurry {img_name}")
                    os.remove(img_path)
                    continue
                img = cv2.resize(img, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                save_path = os.path.join(processed_folder, img_name)

                cv2.imwrite(save_path, img)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue


preprocessing(settings.RAW_DATA_PATH)
# preprocessing(settings.CUSTOM_DATA_PATH)
# print(settings.APP_NAME)
