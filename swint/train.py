import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from models import get_teacher, get_student
from utils import student_loss
from dataloader import get_trainloader, get_testloader

import numpy as np
import random

def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def eval_net(net, loader, batch_size):
	net.eval()
	n_val = len(loader)
	#print(n_val)
	correct = 0
	with tqdm(total=n_val, desc='Validation round', position=0, unit = 'batch', leave=True) as pbar:
		for i, data in enumerate(loader):
			#print(i)
			with torch.no_grad():
				inputs = data[0].to(device = device, dtype=torch.float32)
				#print('-------------------------------',inputs.shape)
				labels = data[1].to(device)
				#print('data-test')
				#print(len(data), len(data[0].shape), len(data[1].shape))
				out = net(inputs.to(device))#.permute(0, 2, 1, 3, 4) # 0 2 1 3 4 for STAM
				#print('out-test')
				#pred = torch.argmax(out, dim=1).to(device)
				_, pred = out.topk(5, 1)
				#print('topk-test')
				#print('@!!!@$!')
				# print(labels.data[0], pred.data[0])
				pred = pred.t()
				labels = labels.view(1, -1).expand_as(pred)
				correct += (pred == labels).to(torch.float32).sum()
				#print('correct-test')
				#print(f'${correct}')
				# print(correct)
			pbar.update()
	#print(f"Validation Loss: {running_loss}")
	acc = correct / (len(loader)*batch_size)
	#print(100 * acc)
	return acc

def train(net, train_loader, val_loader,
		epochs, optimiser, batch_size=None, 
		scheduler= None, 
		teacher=None, ES=True, num_classes=400):
	#criterion_data = nn.BCEWithLogitsLoss(); #add knowledge distillation loss
	losses = []
	val =[]
	accuracies = []
	val_losses = []
	best = 0.0
	early_stopping = 0
	correct = 0
	for epoch in range(epochs):
		net.train()

		running_loss = 0
		i = 0
		with tqdm(total=len(train_loader), desc = "Training epoch: {}".format(epoch),
				position=0, unit = 'batch', leave=True) as pbar:
			for _, data in enumerate(train_loader):
				optimiser.zero_grad()
				#data inputs: N C T H W
				inputs = data[0].to(device = device, dtype=torch.float32)
				labels = data[1].to(device)
				one_hot = F.one_hot(labels, num_classes=400).to(torch.float32)
				# print(inputs.shape,'shapwewww')
				out = net(inputs.to(device))#.permute(0, 2, 1, 3, 4))
				out_teacher = teacher_model(inputs)
				# target = torch.zeros(len(labels), 10).scatter_(1, labels.unsqueeze(1), 1.).to(device = device, dtype=torch.float32)
				loss = student_loss(out, out_teacher, one_hot, tau=4, alpha=0.9)
				running_loss += loss.item()
				i += 1
				loss.backward()
				# for p in net.parameters():
				# 	print(p.grad.norm())
				optimiser.step()
				pbar.update()
				# print(loss)

				
				#print('correct-test')
				#print(f'${correct}')
				# print(correct)
			# pbar.update()
	#print(f"Validation Loss: {running_loss}")
		# acc = correct / (len(loader)*batch_size)

		acc = eval_net(net, val_loader, batch_size)
		torch.optim.lr_scheduler.StepLR(optimiser, step_size=15, gamma=0.2)
		#val_losses.append(val_loss)
		accuracies.append(acc)
		#print('wewew')
		print(f"Validation accuracy {acc}")
		if acc > best: 
			best = acc
			torch.save(net.state_dict(), f"checkpoint3/{root}_{num_classes}_{acc}_{epoch}.pt")
			early_stopping = 0
		else:
			early_stopping += 1
			if early_stopping * ES > 10:
				return
		print(f"\n Epoch {epoch}: Loss {running_loss/(i*batch_size)}")
		losses.append(running_loss/i)

if __name__ == "__main__":
	fixed(69)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	device = torch.device(device)
	teacher_model = get_teacher(400, device)
	student = get_student(224, "resnet50").to(device)
	opt = torch.optim.AdamW(student.parameters(), lr=5e-2)
	#scheduler = torch.optim.scheduler
	batch=16
	root = 'dataset_400'
	train_loader, val_loader = get_trainloader(root=f'../{root}', 
	frames_per_clip = 16, step_between_clips=16,
	num_classes=400, batch_size=batch)

	# torch.save(train_loader, "./train_loader_12.pth")
	# torch.save(val_loader, "./val_loader_12.pth")
	
	# train_loader = torch.load("./train_loader.pth")
	# val_loader = torch.load("./val_loader.pth")
	# print("data loaded!")

	train(student, train_loader, val_loader, 200, opt, batch_size=batch)

