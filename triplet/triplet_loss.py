# reference: https://github.com/adambielski/siamese-triplet/blob/master/losses.py

# triplet loss의 경우 L(a, p, n) = max(0, [ ||V_a - V_p|| - ||V_a - V_n|| + margin ] ) 으로 정의됩니다.
# 직관적으로 설명하면 같은 class인 positive embedding은 기준점인 anchor embedding과 더 가깝게 만들고(+), 반대로 negative embedding과는 더 멀게 만드는(-) loss 인데요,
# 사실 두 거리(a-p, a-n)를 구할 때는 matrix 연산으로 해도 되지만, 제가 했을 때는 그러면 gpu memory 에러가 나서.. 벡터 와이즈 버전으로 코드를 새로 짰습니다.

# 이 코드를 돌리시려면 우선 모든 벡터에 대해 임베딩한 값과, pos, neg mapping에 대한 전처리가 필요합니다. 사실 이 부분이 loss 구하는 것 보다 조금 번거롭습니다.
# input 을 우선 정의해드리겠습니다. 
# anchor: 한 이미지 속 each pixel 에 대한 임베딩 값입니다. 차원은 batch X Channel(e.g., 3 == rgb) X Height X Width 입니다.
# pos_map, neg_map: batch X HW 의 차원을 가지는 dictionary입니다(여기서 HW란 H X W 차원의 이미지를 H*W size를 가지는 벡터로 resize했을때의 크기인 H*W을 의미합니다). 
# key가 엥커를 의미하고, 각 key의 value는 positive/negative class인 픽셀의 위치값 입니다.

# 예를 들어, k번째 픽셀이 3번째와는 pos, 9번째와는 neg 관계를 가지고있다고 가정하면 저희는 v_k = anchor[k], v_p = anchor[3], v_n = anchor[9]을 가져와야합니다.
# 이 때 pos_map[k] = 3, pos_neg[k] = 9 여야 할 것이구요.
# 그런데 픽셀 하나가 단순히 pos, neg관계가 하나만 가지고 있을 리 만무하겠죠? 그래서 randomly 가져오게 되는데요, get_anchor_tag가 해당작업을 해주게 됩니다.

# get_anchor_tag 를 하려면 또 각 픽셀이 어느 클래스 인지에 대한 태깅이 필요한데요, 이게 또 시간이 좀 걸립니다.
# 그래서 전처리로 모든 이미지에 대한 정보를 .npy형식으로 저장해두고, 필요할때 npy 파일을 로드해서 읽으면 시간이 훨씬 덜 걸리게 됩니다.
# 해당 부분은 preprocess.py 를 참고하시되, 현재의 목표에 맞는 방식으로 수정하시고 쓰시면 좋을 것 같습니다.


import numpy as np
import torch
import os


# load tags => preprocess 에서 만든걸 가져옵니다.
npy_path = '/home/userC/soyoungyang/rgb_256_colorization_chi/tags'
btags_train = np.load(os.path.join(npy_path, 'train_b.npy'), allow_pickle=True)
btags_test = np.load(os.path.join(npy_path, 'test_b.npy'), allow_pickle=True)


def get_anchor_tag(filename, phase):
    '''
    randomly make pos, neg tags with filename's anchor
    '''

    filename = filename[0]
    if phase == 'train':
        labels = btags_train.item()
        labels = labels.get(filename)
    elif phase == 'test':
        labels = btags_test.item().get(filename)

    for i, pos in enumerate(labels):
        # get negative labels
        neg_list = [label for j, label in enumerate(labels) if j != i]
        # flatten neg_list
        neg = [l for label in neg_list for l in label]

        len_neg = len(neg)
        len_pos = len(pos)

        pos_map = {}    # dictionary 형태를 사용해서, key 가 anchor의 index, value 가 positive class의 value가 되도록 했습니다.
        neg_map = {}

        for l in pos:
            pos_index = np.random.randint(low=0, high=len_pos)
            if len_pos == 1:
                pos_index = 0
            neg_index = np.random.randint(low=0, high=len_neg - 1)

            pos_map[l] = pos[pos_index]
            neg_map[l] = neg[neg_index]

    return pos_map, neg_map


def triplet_loss(anchor, pos_map, neg_map, margin):
    '''anchor, pos_mapping, neg_mapping all should be TENSOR.'''
    # anchor: 1 X C X H X W    where 1 is Batch size

    anchor = anchor.view(anchor.size(1), -1)        # anchor: C X HW    where c is latent dimension.

    c = anchor.size(0)
    hw = anchor.size(1)

    total_loss = torch.zeros([hw, 1])
    for i in range(hw):

        # get embedding values
        v_a = anchor.narrow_copy(0, i, c)       # V_a : 1 x c

        vector_map_p = np.zeros((hw, 1))
        vector_map_n = np.zeros((hw, 1))
        p = pos_map[i]
        n = neg_map[i]
        vector_map_p[i][p] = 1.                 
        vector_map_n[i][n] = 1.

        vector_map_p = torch.from_numpy(vector_map_p)
        vector_map_n = torch.from_numpy(vector_map_n)

        v_p = torch.mm(vector_map_p, anchor)    # V_p : 1 x C       # 지금보니 이렇게 one hot vector X matrix 형식이 아니라 인덱싱으로 가져와도 되겠네요...^^
        v_n = torch.mm(vector_map_n, anchor)    # V_n : 1 x C

        # get L2 distance
        pos_loss = (v_a - v_p).pow(2).sum(1).pow(.5)
        neg_loss = (v_a - v_n).pow(2).sum(1).pow(.5)

        pos_loss = torch.mean(pos_loss, 1)
        neg_loss = torch.mean(neg_loss, 1)

        # remove the negative values
        clamp = torch.clamp(pos_loss + -1 * neg_loss + margin, min=0)
        # mean for batch
        trip = torch.mean(clamp, 0)

        total_loss[i] = trip

    return total_loss.mean()
