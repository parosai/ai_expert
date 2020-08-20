# ai_expert

* mark down(md) 형식으로 다양하게 사용이 가능하며, 구글에 mark down table/grammar 라고 치시면 다양한 활용법이 나옵니다. [참고](https://heropy.blog/2017/09/30/markdown/)


## TODO

(예시, 보시고 삭제하셔도 돼요~)
- [ ] vgg
- [ ] something ...



## 0820 meeting-related materials

1. Visualization
    - t-SNE
    - UMAP
    - Plotly : https://plotly.com/
    - Confusion matrix via plotly : [link](https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap)

2. Pytorch Codes
    - 공유된 코드 중에 얼마나 좋으냐는 사실 깃헙 레포의 Star 수를 보면 알 수 있습니다. 스타 수가 많은 만큼 많은 사람이 인정하고, 활용하고 있다고 보시면 됩니다.

    - [VGG on cifar10 dataset](https://github.com/kuangliu/pytorch-cifar): 
    cifar10 이라는 간단한 데이터셋에 대해 여러 모델(VGG, ResNet, DenseNet 등)을 models 라는 하위 폴더에 각각의 class로 구현하여 다양한 모델을 체험(?) 및 활용해볼 수 있는 레포입니다.
    이미 구현되어있는 모델(torchvision.models.vgg19)을 쓰셔도 좋지만, 
    이렇게 직접 코드를 다뤄보시면 각 레이어가 어떻게 구성되어있는지 파악하시기 좋고, 혹은 직접 레이어의 조건을 바꾼다거나, 특정 레이어를 꺼내온다거나 하는 등의 자유도가 높아지게 됩니다.
    이 코드는 cifar10 데이터셋에서 정말 유명한 코드로 보시면 됩니다. ㅎㅎ
    예를 들어, 저희가 이 모델을 가져온다면 마지막 classifier layer의 node 갯수만 10 -> 7 로 바꾸면 될 것 같습니다.

    - [Pytorch tutorial](https://github.com/yunjey/pytorch-tutorial) (optional): 
    딥러닝을 계속 공부하실 계획이라면 파이토치를 추천드리고, 토치를 시작하는 분들을 위한 pytorch tutorial 입니다.
    코드와 코멘트가 매우 깔끔하고 정돈되어 있으니, 토치를 쓰실 때 참고하시기 좋습니다.
  
3. 코멘트
    - unsupervised setting(e.g., Triplet Loss)에 대해서는 며칠 후에 공유드리도록 하겠습니다.
    - 시간 상관없이 편안하게 슬렉에 질문이나 문제상황 올려주시면 저희가 보는대로 답변 도와드리겠습니다 :) 
