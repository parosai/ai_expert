import torch
import torch.nn as nn
import torch.nn.functional as F

import typing

class MyModule(nn.Module):

    def __init__(self, inDim: int = 64, outDim: int = 64) -> None:

        # Python 에서 class 상속에 필요한 문법 (부모 클래스의 생성자를 호출)
        super(MyModule, self).__init__()

        self.linear = nn.Linear(inDim, outDim)
        self.norm = nn.BatchNorm1d(outDim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = x

        out = self.linear(out)
        out = F.relu(out)
        out = self.norm(out)

        return out

class MyModel(nn.Module):

    def __init__(self, inDim: int = 16, hiddenDim: int = 64, outDim: int = 10
                numLayers: int = 10) -> None:

        super(MyModel, self).__init__()

        self.layers = [MyModule(inDim, hiddenDim)]

        # ResNet 에서의 resiual block 같이 비슷한 구조가 여러 번 반복되는 경우에 이렇게 코드를 구성할 수 있어요
        for _ in range(numLayers):

            self.layers.append(MyModule(hiddenDim, hiddenDim))

        self.layers.append(MyModule(hiddenDim, outDim))

        # nn.ModuleList 로 감싸줘야 torch 에서 레이어의 존재를 감지합니다
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = x

        for idx, layer in enumerate(self.layers):

            out = layer(out)
            # ResNet 에서의 skip-connection 은
            # out = layer(out) + out
            # 처럼 짤 수 있습니다 (물론 여기서는 에러 발생)

        out = F.log_softmax(out)

        return out