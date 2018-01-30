import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from utils.image_processing import rgb2gray
from utils.datasets import DatasetManager, split_data, load_npz

dataset_dir = os.path.join(os.getcwd(), 'ml_dataset', 'datasets')

# dataset_name = argv[1]
# dataset_dir = argv[2]

# input_size = (128, 128)
# use_gray_scale = True

# def load_img_data(paths, num_classes, input_size, use_gray=True):
#     nclass     = num_classes
#     valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
#     imgcnt     = 0
#     for i, relpath in zip(range(nclass), paths):
#         path = cwd + "/" + relpath
#         flist = os.listdir(path)
#         for f in flist:
#             if os.path.splitext(f)[1].lower() not in valid_exts:
#                 continue
#             fullpath = os.path.join(path, f)
#             currimg  = imread(fullpath)
#             # Convert to grayscale  
#             if use_gray:
#                 grayimg  = rgb2gray(currimg)
#             else:
#                 grayimg  = currimg
#             # Reshape
#             graysmall = imresize(grayimg, [input_size[0], input_size[1]])/255.
#             grayvec   = np.reshape(graysmall, (1, -1))
#             # Save 
#             curr_label = np.eye(nclass, nclass)[i:i+1, :]
#             if imgcnt is 0:
#                 totalimg   = grayvec
#                 totallabel = curr_label
#             else:
#                 totalimg   = np.concatenate((totalimg, grayvec), axis=0)
#                 totallabel = np.concatenate((totallabel, curr_label), axis=0)
#             imgcnt    = imgcnt + 1
#     print ("Total %d images loaded." % (imgcnt))

def divide_dataset(x_data, y_data, train_rate=0.6, valid_rate=0.2, test_rate=0.2):
    if train_rate + valid_rate + test_rate != 1:
        print("train_rate, valid_rate, test_rate 세 값의 총합이 반드시 1이 되어야 합니다.")
        return
    total_cnt = len(y_data)
    randidx = np.random.randint(total_cnt, size=total_cnt)
    trainidx = randidx[0:int(total_cnt * train_rate)]
    valididx = randidx[int(total_cnt * train_rate):int(total_cnt * (train_rate+valid_rate))]
    testidx = randidx[int(total_cnt * (train_rate+valid_rate)):total_cnt]
    x_train = x_data[trainidx, :]
    y_train = y_data[trainidx, :]
    x_valid = x_data[valididx, :]
    y_valid = y_data[valididx, :]
    x_test = x_data[testidx, :]
    y_test = y_data[testidx, :]

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def save_dataset(save_path, x_data, y_data, data_name='custom_dataset'):
    inputsize = x_data[0].shape

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset_file = data_name + ".npz"
    savepath = os.path.join(save_path, dataset_file)

    train_data, valid_data = split_data(x_data, y_data, .3)
    x_train, y_train = train_data
    val_data, test_data = split_data(valid_data[0], valid_data[1], .4)
    x_valid, y_valid = val_data
    x_test, y_test = test_data

    #x_train, y_train, x_valid, y_valid, x_test, y_test = divide_dataset(x_data, y_data)

    np.savez(savepath, x_train=x_train, y_train=y_train,
                        x_valid=x_valid, y_valid=y_valid,
                        x_test=x_test, y_test=y_test, inputsize=inputsize)
    print ("Saved to %s" % (savepath))

input_size = (128, 128)
data_loader = DatasetManager('fer2013', input_size)
x_data, y_data = data_loader.get_data()
save_dataset(dataset_dir, x_data, y_data, 'fer2013')