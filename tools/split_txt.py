import os
from random import shuffle
from math import floor
import glob

def main():
    all_files = os.listdir("C:\\Users\\Darkhan\\source\\repos\\master-thesis-equipment-detection-docs\\pumps\\dataset_name_rendered\\JPEGImages\\")
    shuffle(all_files)
    split = 1
    split_index = int(floor(len(all_files) * split))
    training = all_files[:split_index]
    testing = all_files[split_index:]

    traival_files_wr = [x.split('.')[0] + '\n' for x in training]
    test_files_wr = [x.split('.')[0] + '\n' for x in testing]
    with open(os.path.join("C:\\Users\\Darkhan\\source\\repos\\master-thesis-equipment-detection-docs\\pumps\\dataset_name_rendered\\ImageSets\\Main", 'trainval.txt'), 'w') as f:
        f.writelines(traival_files_wr)

    with open(os.path.join("C:\\Users\\Darkhan\\source\\repos\\master-thesis-equipment-detection-docs\\pumps\\dataset_name_rendered\\ImageSets\\Main", 'test.txt'), 'w') as f:
        f.writelines(test_files_wr)

    print("Success")

if __name__ == '__main__':
    main()