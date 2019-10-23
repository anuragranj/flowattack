#!/usr/bin/env python2.7

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch
from torch.nn import init



class Preprocess(torch.nn.Module):
	def __init__(self, pre_normalization):
		super(Preprocess, self).__init__()
		self.pre_normalization = pre_normalization
	# end

	def forward(self, variableInput):
		variableRed = variableInput[:, 0:1, :, :]
		variableGreen = variableInput[:, 1:2, :, :]
		variableBlue = variableInput[:, 2:3, :, :]

		if self.pre_normalization is not None:
			if hasattr(self.pre_normalization, 'mean') and hasattr(self.pre_normalization, 'std'):
				_mean = self.pre_normalization.mean
				_std = self.pre_normalization.std
			else:
				_mean = variableInput.transpose(0,1).contiguous().view(3, -1).mean(1)
				_std = variableInput.transpose(0,1).contiguous().view(3, -1).std(1)

			variableRed = variableRed * _std[0]
			variableGreen = variableGreen * _std[1]
			variableBlue = variableBlue * _std[2]

			variableRed = variableRed + _mean[0]
			variableGreen = variableGreen + _mean[1]
			variableBlue = variableBlue + _mean[2]

		variableRed = variableRed - 0.485
		variableGreen = variableGreen - 0.456
		variableBlue = variableBlue - 0.406

		variableRed = variableRed / 0.229
		variableGreen = variableGreen / 0.224
		variableBlue = variableBlue / 0.225

		return torch.cat([variableRed, variableGreen, variableBlue], 1)

class Basic(torch.nn.Module):
	def __init__(self, intLevel, arguments_strModel):
		super(Basic, self).__init__()
		self.intLevel = intLevel

		self.moduleBasic = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
		)

		if intLevel == 5:
			if arguments_strModel == '3' or arguments_strModel == '4':
				intLevel = 4 # the models trained on the flying chairs dataset do not come with weights for the sixth layer

		for intConv in range(5):
			self.moduleBasic[intConv * 2].weight.data.copy_(torch.load('models/spynet_models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-weight.pth.tar'))
			self.moduleBasic[intConv * 2].bias.data.copy_(torch.load('models/spynet_models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-bias.pth.tar'))

	def forward(self, variableInput):
		return self.moduleBasic(variableInput)

class Backward(torch.nn.Module):
	def __init__(self):
		super(Backward, self).__init__()

	def forward(self, variableInput, variableFlow):
		if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != variableInput.size(0) or self.tensorGrid.size(2) != variableInput.size(2) or self.tensorGrid.size(3) != variableInput.size(3):
			torchHorizontal = torch.linspace(-1.0, 1.0, variableInput.size(3)).view(1, 1, 1, variableInput.size(3)).expand(variableInput.size(0), 1, variableInput.size(2), variableInput.size(3))
			torchVertical = torch.linspace(-1.0, 1.0, variableInput.size(2)).view(1, 1, variableInput.size(2), 1).expand(variableInput.size(0), 1, variableInput.size(2), variableInput.size(3))

			self.tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda()
		# end

		variableFlow = torch.cat([ variableFlow[:, 0:1, :, :] / ((variableInput.size(3) - 1.0) / 2.0), variableFlow[:, 1:2, :, :] / ((variableInput.size(2) - 1.0) / 2.0) ], 1)

		variableGrid = torch.autograd.Variable(data=self.tensorGrid) + variableFlow

		return torch.nn.functional.grid_sample(input=variableInput, grid=variableGrid.clamp(-1.0, 1.0).permute(0, 2, 3, 1), mode='bilinear')

class Network(torch.nn.Module):
	"""
	Creates SpyNet model for estimating optical flow.
	If images passed
	TODO:
	"""
	def __init__(self, nlevels, strmodel='F', pre_normalization=None, pretrained=True):
		super(Network, self).__init__()
		print('Creating Spynet with', nlevels, 'levels')
		self.nlevels = nlevels
		self.strmodel = strmodel
		self.pre_normalization = pre_normalization
		self.pretrained = pretrained

		self.modulePreprocess = Preprocess(pre_normalization=pre_normalization)
		self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel, strmodel) for intLevel in range(nlevels) ])
		self.moduleBackward = Backward()

		if not self.pretrained:
			for m in self.modules():
				if isinstance(m, torch.nn.Conv2d):
					if m.bias is not None:
						init.uniform(m.bias)
					init.xavier_uniform(m.weight)


	def forward(self, variableFirst, variableSecond):
		variableAllFlows = [ 0 for i in range(self.nlevels)]

		variableFirst = [ self.modulePreprocess(variableFirst) ]
		variableSecond = [ self.modulePreprocess(variableSecond) ]

		for intLevel in range(self.nlevels-1):
			#if variableFirst[0].size(2) > 32 or variableFirst[0].size(3) > 32:
			#	print('downsample', intLevel)
			variableFirst.insert(0, torch.nn.functional.avg_pool2d(input=variableFirst[0], kernel_size=2, stride=2))
			variableSecond.insert(0, torch.nn.functional.avg_pool2d(input=variableSecond[0], kernel_size=2, stride=2))
			# end
		# end

		variableFlow = torch.autograd.Variable(data=torch.zeros(variableFirst[0].size(0), 2, int(math.floor(variableFirst[0].size(2) / 2.0)), int(math.floor(variableFirst[0].size(3) / 2.0))).cuda())

		for intLevel in range(len(variableFirst)):
			variableUpsampled = torch.nn.functional.upsample(input=variableFlow, scale_factor=2, mode='bilinear') * 2.0

			if variableUpsampled.size(2) != variableFirst[intLevel].size(2): variableUpsampled = torch.nn.functional.pad(variableUpsampled, [0, 0, 0, 1], 'replicate')
			if variableUpsampled.size(3) != variableFirst[intLevel].size(3): variableUpsampled = torch.nn.functional.pad(variableUpsampled, [0, 1, 0, 0], 'replicate')

			variableFlow = self.moduleBasic[intLevel](torch.cat([ variableFirst[intLevel], self.moduleBackward(variableSecond[intLevel], variableUpsampled), variableUpsampled ], 1)) + variableUpsampled
			variableAllFlows[self.nlevels-intLevel-1] = variableFlow
		# end
		if self.training:
			return variableAllFlows
		else:
			return variableFlow
