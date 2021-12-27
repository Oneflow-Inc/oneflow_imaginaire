import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(ex) for ex in IMG_EXTENSIONS)

def make_dataset(dir):
    images_root = []
    assert os.path.isdir(dir), str(dir) + 'is not a valid directory'

    for root, _, file_names in sorted(os.walk(dir)):
        for file_name in file_names:
            if is_image_file(file_name):
                path = os.path.join(root, file_name)
                images_root.append(path)

    return images_root