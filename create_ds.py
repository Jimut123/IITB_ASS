
import glob
import shutil
from tqdm import tqdm

"""
all_images = glob.glob('Kolkata_020/*')
for item in tqdm(all_images[:10000]):
    name = item.split('/')[-1]
    if name == 'train':
        continue
    path = "Kolkata_020/"
    moveto = "Kolkata_020/train/"
    src = path+name
    dst = moveto+name
    shutil.move(src,dst)

for item in tqdm(all_images[10000:]):
    name = item.split('/')[-1]
    if name == 'test' or name == 'train':
        continue
    path = "Kolkata_020/"
    moveto = "Kolkata_020/test/"
    src = path+name
    dst = moveto+name
    shutil.move(src,dst)
"""

print(len(glob.glob('Kolkata_020/test/*')))
print(len(glob.glob('Kolkata_020/train/*')))



