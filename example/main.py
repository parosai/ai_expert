import os
import sys
import timeit
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

# 다른 파일에서 구현한 클래스를 불러와서 사용합니다
import model as m
import dataloader as dl

def main(hParam):

	# 학습 때의 여러 정보를 기록하는 용도
	# logger 나 tensorboard 등을 사용하는 경우도 있습니다
	log = {'loss': list(), 'time': list()}

	# GPU 가 사용할 수 있는지 확인합니다
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model = m.MyModel(inDim = hParam['inDim'], hiddenDim = hParam['hiddenDim'],
					outDim = hParam['outDim'], numLayers = hParam['numLayers'])
	model = model.to(device)

	# 경우에 따라 train / test data 를 따로 불러와야하지만 여기서는 하나로 통일합니다
	# GPU 가 연산하는 동안 CPU 에서는 다음 이터레이션에 필요한 데이터를 불러오는데요,
	# 데이터를 준비하는 데에 사용할 CPU 의 개수가 num_workers 입니다
	# 이 값은 너무 적어도, 너무 많아도 성능에 악영향을 줍니다
	data = dl.ExampleData(hParam['data_path'], hParam['option'])
	loader = torch.utils.data.DataLoader(data, batch_size = hParam['batch'],
											num_workers = 4, pin_memory = True, shuffle = True)

	# F.log_softmax 에 알맞은 loss function 은 NLLLoss 입니다
	nll = nn.NLLLoss()
	# Adam optimizer 를 사용합니다
	# 어떤 파라미터를 최적화하고 싶은지 optimizer 에게 알려주어야 하는데요, list(model.parameters()) 를 넘겨주면 됩니다
	# 만약 전체 모델의 일부만 최적화 (finetuning 등) 하고 싶다면, 위 list 내부의 값을 조정해주면 됩니다
	# 혹은 파라미터별로 다른 learning rate 를 적용하고 싶다면, 아래와 같이 해주시면 됩니다
	optimizer = torch.optim.Adam(list(model.parameters()), lr = hParam['lr'])
	"""
	optimzer.add_param_group(
		{
			'params': model.parameters() 의 일부,
			'lr': 0.01,
			...
		}
	) 를 필요에 따라 적절하게 설정
	"""
	# 아래와 같이 learning rate scheduler 를 사용할 수도 있습니다
	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, ...)

	# 모델을 학습 모드로 설정합니다
	# batch norm, dropout 등은 학습 / 테스트 시에 다르게 작동하기 때문에, 꼭 확인해줘야 합니다
	model.train()

	for epoch in range(hParam['epoch']):

		# epoch 당 시간 측정
		timeNow = timeit.default_timer()

		# tqdm 같은 library 를 사용해도 됩니다
		for idx, (txt, label) in enumerate(loader)

			txt, label = txt.to(device), label.to(device)

			pred = model(txt)
			loss = nll(pred, label)

			# Pytorch 에서 loss 에서 계산된 gradient 는 계속 누적됩니다
			# 그 이유는 배치를 여러 번 나누어 gradient 를 계산하고 이를 누적해 사용하면,
			# full-batch gradient descent 를 해줄 수 있기 때문입니다
			# 만약 mini-batch gradient descent 를 하거나 새로운 epoch 이 되면, 누적된 gradient 를 없애야 합니다
			# optimizer.zero_grad() 를 사용해 누적된 gradient 를 0 으로 초기화합니다
			optimizer.zero_grad()
			# loss 를 미분해줍니다
			loss.backward()
			# 지금까지 계산된 gradient 를 가지고 모델의 파라미터를 업데이트해줍니다
			optimizer.step()
			# 만약 loss 를 여러 개 사용하면, loss 를 서로 더해주거나 각 loss 마다 backward 를 해줍니다
			# 예시 1)
			# optimizer.zero_grad()
			# total_loss = loss1 + loss2 + loss3
			# total_loss.backward()
			# optimizer.step()
			# 예시 2)
			# optimizer.zero_grad()
			# loss1.backward()
			# loss2.backward()
			# loss3.backward()
			# optimizer.step()

		scheduler.step()

		log['loss'].append(loss.item())
		log['time'].append(timeit.default_timer() - timeNow)

		if hParam['verbose']:

			# 진행 상황 출력
			print('[info] Epoch : [{}/{}], Loss : {}'.format(epoch + 1, hParam['epoch'], log['loss'][-1]))
			print('[info] Time : {}'.format(log['time'][-1]))

	# 모델을 테스트 모드로 설정합니다
	model.eval()

	# 테스트 시에는 gradient 계산이 필요 없으므로 설정해줍니다
	with torch.no_grad():

		for epoch in range(hParam['epoch']):

			for idx, (txt, label) in enumerate(loader)

				txt, label = txt.to(device), label.to(device)

				pred = model(txt)

				# pred 를 이용해 하고싶은 일을 합니다
				# 예시 1) 예측 결과를 numpy array 로 변경
				# pred = pred.cpu().numpy()
				# 예시 2) 정답과 일치한 횟수를 세기 -> label = [1, 1, 0, 1], pred = [0, 1, 0, 0]
				# (label == pred).sum() ----> [False, True, True, False] ----> 2

	# 학습 / 테스트 종료 및 결과 저장
	print('Train finished')

	# 모델 파라미터 저장
	torch.save(model.state_dict(), hParam['model_path'])

	# log 저장 (pickle)
	with open(hParam['log_path'], 'wb') as fs:

		pickle.dump(log, fs)

def train(model, loader)



# Python 에서의 main 에 해당하는 부분입니다
if __name__ == '__main__':

	# 여러 학습 세팅을 불러옵니다
	# 대다수의 사람들은 json 이나 yaml 을 쓰지만,
	# 저는 py 파일에 dict 를 선언하고 거기에 설정 값을 넣습니다
	with open(sys.argv[1], 'r') as fs:

		raw = fs.read()
		exec(raw)

	main(hParam)