from imutils import paths
import random
import os
import shutil
# /Users/mandywoo/Documents/cnn_projects/fruits360-kaggle/dataset/fruits-360/Test/Apple Braeburn/3_100.jpg
# /Users/mandywoo/Documents/cnn_projects/fruits360-kaggle/dataset/background/ukbench00000.jpg
# /Users/mandywoo/Documents/cnn_projects/fruits360-kaggle/dataset/fruits-360/Backgrounds
background_image_paths = sorted(list(paths.list_images('fruits360-kaggle/dataset/background')))
to_delete = random.sample(range(len(background_image_paths)), 500)
train_background_image_paths = [background_image_paths.pop(i) for i, _ in reversed(list(enumerate(background_image_paths))) if i in to_delete]
to_delete = random.sample(range(len(background_image_paths)), 500)
test_background_image_paths = [background_image_paths.pop(i) for i, _ in reversed(list(enumerate(background_image_paths))) if i in to_delete]


for path in train_background_image_paths:
    shutil.move('/Users/mandywoo/Documents/cnn_projects/' + path, '/Users/mandywoo/Documents/cnn_projects/fruits360-kaggle/dataset/fruits-360/Backgrounds/background/'+path.split(os.path.sep)[-1])