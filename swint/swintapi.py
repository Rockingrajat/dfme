from mmaction.core import OutputHook
from mmcv.parallel import scatter
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer 
import torch

class SwinT():
	def __init__(self, device):
		super().__init__()
		config_file = '../../Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py' #for swin-T
		checkpoint_file = '/home/SharedData/rohan/interiit/datafree-model-extraction/dfme/checkpoint/swinT/swin_tiny_patch244_window877_kinetics400_1k.pth' #change this for other env
		self.device = torch.device(device)
		self.model = init_recognizer(config_file, checkpoint_file, device=device)
		self.model.eval()

	def SwinOneVid(self, data, as_tensor= True):
		'''data is a tensor: nxtxcxhxw'''
		data = data.permute(0,2,1,3,4).unsqueeze(0)
		if next(self.model.parameters()).is_cuda:
		# scatter to specified GPU
			data = scatter(data, [self.device])[0]
		# forward the model
		with OutputHook(self.model, outputs=True, as_tensor=as_tensor) as h:
			with torch.no_grad():
				scores = self.model(data,return_loss=False)[0]
			return torch.from_numpy(scores).to(self.device)

	def Swinbatch(self, x):
		pred_victim_pts = torch.zeros(N,args.num_classes)
		for i in range(N):
		    pred_victim_pts[i] = SwinOneVid(x_[i].unsqueeze(0))  
		return pred_victim_pts

	def __call__(self, x):
		ans = self.Swinbatch(x)
		return ans