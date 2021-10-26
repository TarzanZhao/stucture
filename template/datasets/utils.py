import os
import json

tar_path = "/com_space/zhaohexu/cityscapes/gtFine"
input_path = "/com_space/zhaohexu/cityscapes/leftImg8bit"

folders = ['val', 'train', 'test']
all_cities = []
for folder in folders:
    path = os.path.join(input_path, folder)
    all_cities += [ os.path.join(folder, file) for file in os.listdir(path)]
# print(all_cities)
with open("cities.json", "w") as f:
    json.dump(all_cities, f)