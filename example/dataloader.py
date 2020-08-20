import os
import sys
import random
import numpy as np

import torch
from torch.utils.data import Dataset

class ExampleData(Dataset):

	def __init__(self, filepath: str, option: bool = True):

		# 여러 줄로 되어있는 텍스트 파일을 데이터셋으로 사용한다고 가정
		# 예시)
		# 뽀로로, 펭귄
		# 둘리, 공룡
		# 펭수, 펭귄
		# ...
		# 이미지 데이터의 경우에는 PIL 등으로 열어야 합니다
		# 한편, 파일 여러 개로 구성되는 데이터셋의 경우에는 파일의 경로만 불러와놓고,
		# 아래의 __getitem__ 에서 해당 파일을 열어 사용하는 경우가 많습니다
		with open(filepath, 'r') as fs:

			self.data = fs.readlines()

		if option:

			# 옵션에 따라서 여러 데이터 전처리를 해줄 수 있음
			# 예시) train data 와 test data 를 분리
			pass

		else:

			pass

	def __len__(self):

		# 데이터셋의 전체 길이를 제공해줘야 합니다
		return len(self.data)

	def __getitem__(self, idx):

		# 해당 idx 에 해당하는 데이터와 정답을 반환하는 방식으로 구현합니다
		# 실제로는 torch.from_numpy() 등을 이용해 torch.Tensor 로 변환한 데이터이어야 합니다
		sample = self.data[idx]
		txt, label = data.split(',')

		return txt, label

# 다른 방법: __iter__ 에서 yield 를 사용하는 방식으로 구현하는 경우도 있습니다
# __getitem__ 을 사용하는 경우는 보통 한 에폭에 모든 데이터를 한 번씩 보고 지나갈 때,
# __iter__ 를 사용하는 경우는 모든 데이터를 한 번씩 본다고 보장하진 못하지만, 그래도 정해진 횟수만큼 데이터를 보고 싶을 때입니다
"""
	def __iter__(self):

		while True:

			sample = self.data[random.randint(0, len(self.data) - 1)]
			txt, label = sample.split(',')

			yield txt, label
"""