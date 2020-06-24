import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import models
import torch.nn as nn
from fastai.vision import Path
import torch
from torch.autograd import Variable
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import zipfile
from matplotlib.pyplot import imshow
from tqdm import tqdm
from sklearn.metrics import precision_score

ALL_CHAR_SET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 5
batch_size = 64

data = pd.read_csv('data.csv', header=None)

def encode(a):
	onehot = [0]*ALL_CHAR_SET_LEN
	try:
		idx = ALL_CHAR_SET.index(a)
	except:
		return onehot
	onehot[idx] += 1
	return onehot

class Mydataset(Dataset):
	def __init__(self, path, is_train=True, transform=None):
		self.path = path
		if is_train: self.img = os.listdir(self.path)[:30000]
		else: self.img = os.listdir(self.path)[30001:]
		self.transform = transform
		
	def __getitem__(self, idx):
		img_path = self.img[idx]
		img = Image.open(self.path/img_path)
		img = img.convert('L')
		plt.imshow(img)
		path = Path(self.path/img_path).name[:-4]
#         print(path)
		label =data[data[0] == int(path)][1].item()
		label_list = list(label)
		label_list+= ['$']*(MAX_CAPTCHA - len(label_list))
		label = "".join(label_list)
#         print(label)
#         print(label)
		label_oh = []
		for i in label_list:
			label_oh += encode(i)
		if self.transform is not None:
			img = self.transform(img)
		return img, np.array(label_oh), label
	
	def __len__(self):
		return len(self.img)

transform = transforms.Compose([
	transforms.Resize([224, 224]),
	transforms.ToTensor(),
])

train_ds = Mydataset(Path('test-task-data'), transform=transform)
test_ds = Mydataset(Path('test-task-data'), False, transform)
train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=1, num_workers=0)

train_ds.__getitem__(23)[1].shape
# train_ds.__getitem__(1)

model = models.resnet18(pretrained=False)

model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model.fc = nn.Linear(in_features=512, out_features=ALL_CHAR_SET_LEN*MAX_CAPTCHA, bias=True)

ALL_CHAR_SET_LEN*MAX_CAPTCHA

model.cuda();

loss_func = nn.MultiLabelSoftMarginLoss()
optm = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
	for step, i in enumerate(train_dl):
		img, label_oh, label = i
		img = Variable(img).cuda()
		label_oh = Variable(label_oh.float()).cuda()
		pred = model(img)
		if step==0:
			labels_res = []
			c_res = []
			for j in range(batch_size):
				pred_sample = pred[j]
				c0 = ALL_CHAR_SET[np.argmax(pred_sample.squeeze().cpu().tolist()[0:ALL_CHAR_SET_LEN])]
				c1 = ALL_CHAR_SET[np.argmax(pred_sample.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN:ALL_CHAR_SET_LEN*2])]
				c2 = ALL_CHAR_SET[np.argmax(pred_sample.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*2:ALL_CHAR_SET_LEN*3])]
				c3 = ALL_CHAR_SET[np.argmax(pred_sample.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*3:ALL_CHAR_SET_LEN*4])]
				c4 = ALL_CHAR_SET[np.argmax(pred_sample.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*4:ALL_CHAR_SET_LEN*5])]
				c = '%s%s%s%s%s' % (c0, c1, c2, c3, c4)
				c_res.append(c)
				print('prediction: ',epoch+1, " is ", c)
				print('label: ',epoch+1, " is ", label[j])
				labels_res.append(label[j])
			print('presicion epoch', epoch+1, ' is ', precision_score(labels_res, c_res, average='macro'))
		loss = loss_func(pred, label_oh)
		optm.zero_grad()
		loss.backward()
		optm.step()
	print('epoch:', epoch+1, 'loss:', loss.item())

model.eval();

for step, (img, label_oh, label) in enumerate(test_dl):
	img = Variable(img).cuda()
	pred = model(img)
	c0 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[0:ALL_CHAR_SET_LEN])]
	c1 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN:ALL_CHAR_SET_LEN*2])]
	c2 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*2:ALL_CHAR_SET_LEN*3])]
	c3 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*3:ALL_CHAR_SET_LEN*4])]
	c4 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*4:ALL_CHAR_SET_LEN*5])]
	c = '%s%s%s%s%s' % (c0, c1, c2, c3, c4)
	labels_res = []
	print('prediction: ',c)
	print('label: ', label[0])
	labels_res.append(label[0])
	c_res = []
	c_res.append(c)

print('presicion = ', precision_score(labels_res, c_res, average='macro'))

PATH = 'model_only_numbers.pth'
torch.save(model.state_dict(), PATH)
