from os import listdir
from os.path import join
import os

from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import time


#######################################
# save index list of same segmentation
#######################################

def get_tags(img):
    '''
    input: Tensorform image. [3, 256, 256]
    returns:
    1) labels = [[label 1 pixels], [label 2 pixels], ... , [label k pixels]]
    2) num of labels
    '''

    img = img.view(img.size(0), -1)  # [3, 256*256]
    img = np.transpose(img, [1, 0])  # 256*256 x 3
    is_labeled = np.zeros([img.size(0), 1], dtype=int)  # 256*256 x 1
    labels = []

    th = 0.2  # threshold

    for i, val_a in enumerate(img):

        if is_labeled[i] == 1:
            continue

        # exclude white ones
        mean = val_a.mean()
        if mean == 1.0:
            is_labeled[i] = 1
            continue

        label_i = []
        val_a = val_a.unsqueeze(0)

        for j, val_b in enumerate(img):

            if is_labeled[j] == 1:
                continue

            # distance 이하이면 같은 label로 묶어라~ 인데요, 이 부분을 수정하시면 될 것 같습니다.
            val_b = val_b.unsqueeze(0)
            distance = torch.norm((val_a - val_b), 2, -1)

            if distance.item() <= th:
                label_i.append(j)
                is_labeled[j] = 1

        labels.append(label_i)

    return labels


def save_tags_to_np(path, phase, type):
    '''
    save labels of target B as .npy file.
    '''

    if not os.path.exists(join(path, 'tags')):
        os.mkdir(join(path, 'tags'))

    src_path = os.path.join(path, phase)#, type)
    image_filenames = [x for x in listdir(src_path)]

    outputs = {}
    i = 0
    last_time = time.time()

    for filename in image_filenames:
        b = Image.open(os.path.join(src_path, filename))
        b = b.convert("RGB")

        # normalization
        b = b.resize((256, 256), resample=Image.NEAREST)
        b = transforms.ToTensor()(b)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        labels = get_tags(b)
        outputs[str(filename)] = labels

        # 중간중간 확인용으로 print하는 것이니 신경 안쓰셔도 됩니다. 시간이 걸리는지 확인하는 용도로 time.time을 씁니다.
        i+=1
        print("{}\t{}th tags are finished.".format(phase, i))
        if i%30 == 0:
            cur_time = time.time()
            step_time = cur_time - last_time
            last_time = cur_time
            print("time of tagging 30 images: {}".format(step_time))
            print("{}\t{}th tags are finished.".format(phase, i))

    trg_path = os.path.join(path, 'tags')
    np.save(join(trg_path, "{}_{}_2.npy".format(phase, type)), outputs)


def save_tags():

    # path = '/home/userC/soyoungyang/rgb_256_colorization_chi'
    path = '/home/userC/soyoungyang/rgb_256_colorization_chi/hint_scribble'
    save_tags_to_np(path, 'h', 9)
    print('Hint data tags are saved in', path +'/tags')
    # save_tags_to_np(path, 'test', 'b')
    # print('Test data tags are saved in', path +'/tags')

# run
if __name__ == "__main__":
    save_tags()
