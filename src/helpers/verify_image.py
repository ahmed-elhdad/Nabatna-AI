import cv2
from PIL import Image
def is_blurry(image_path, threshold=100):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold
def is_corrupted(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return False
    except Exception:
        return True