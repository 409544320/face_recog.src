# -*- coding: utf-8 -*-
import os, glob
import random

all_folder= "extYaleB/" # ���ݼ�Ŀ¼
model= "extYaleB.SRC" # SRCģ��
n_train= 30 # ÿ���˵�ѵ��ͼ����(���ѡȡ)
train_list= "extYaleB.train_list" # ѵ��ͼ���б�, �豣֤ÿ���˵�ѵ��ͼ����������
test_list= "extYaleB.test_list" # ����ͼ���б�
sci_t= 0 # SCI��ֵ��������ͼ�PAMI-Face.pdf�ļ�

w= 10 # SRCʹ������ͼ����(˫���Բ�ֵ)
h= 12 # SRCʹ������ͼ��߶�

train_cmdline= "train.exe %s %d %d %d %s" % (train_list, n_train, w, h, model) # ѵ��������
test_cmdline= "test.exe %s %s %f" % (model, test_list, sci_t) # ����������

def UnrepeatedIntRandom(lower, upper, count):
	"""Uniformly distributed COUNT integer randoms
	[LOWER, UPPER]
	COUNT must be <= (UPPER-LOWER+1)
	"""
	
	n= upper-lower+1
	if count>n:
		print "[error : UnrepeatedIntRandom], COUNT must be <= (UPPER-LOWER+1)."
		exit(-1)
	x= range(lower, upper+1)
	urdm= []
	for i in range(0, count):
		t= random.randint(i, n-1)
		tmp= x[i]
		x[i]= x[t]
		x[t]= tmp
		urdm.append(x[i])
	return urdm

def Filename2Id(path):
	idx= path.find("yaleB")
	return path[idx:idx+7]

def SplitTrainTest():
	train_f= open(train_list, "w")
	test_f= open(test_list, "w")
	prev_id= ""
	subject_samples= []
	for path in glob.glob(all_folder+"*.pgm"):
		id= Filename2Id(path)
		print id
		if prev_id==id:
			subject_samples.append(path)
		elif prev_id!="":
			train_samples= UnrepeatedIntRandom(0, len(subject_samples)-1, n_train)
			for i in range(0, len(subject_samples)):
				if i in train_samples:
					train_f.write(subject_samples[i])
					train_f.write("\n")
				else:
					test_f.write(subject_samples[i])
					test_f.write("\n")
			prev_id= id
			subject_samples= []
			subject_samples.append(path)
		else:
			prev_id= id
			subject_samples.append(path)
	train_samples= UnrepeatedIntRandom(0, len(subject_samples)-1, n_train)
	for i in range(0, len(subject_samples)):
		if i in train_samples:
			train_f.write(subject_samples[i])
			train_f.write("\n")
		else:
			test_f.write(subject_samples[i])
			test_f.write("\n")
	train_f.close()
	test_f.close()
	
def Train():
	os.system(train_cmdline)
	
def Test():
	os.system(test_cmdline)

if __name__=='__main__':
	SplitTrainTest()
	Train()
	Test()