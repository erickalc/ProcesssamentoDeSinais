#coding: utf- 8
import csv
import numpy as np
import sys
import pdb
import scipy as sp
import mne
from scipy import signal
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#metodo para carregar arquivo edf
#retorna uma matriz numpy. Nas linhas são os canais, nas colunas são os dados
def readEDF(fileName):
	raw = mne.io.read_raw_edf(fileName, preload=True,verbose=True)
	return raw._data

#http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html
def computePSD(mChosenChns):
	f1, psd1 = signal.welch(mChosenChns[0,:],fs=200)
	f2, psd2 = signal.welch(mChosenChns[1,:],fs=200)
	f3, psd3 = signal.welch(mChosenChns[2,:],fs=200)
	f4, psd4 = signal.welch(mChosenChns[3,:],fs=200)
	myArray = np.concatenate((psd1, psd2, psd3, psd4))
	#return psd1, psd2, psd3, psd4
	#pdb.set_trace()
	return myArray
	
def getChosenChannels(edfFileName):
	mEDF = readEDF(edfFileName)
	
	#vamos utilizar dados de apenas 4 canais:
	#fp1 : canal na linha 0
	#fp2 : canal na linha 1
	#f3 : canal na linha 2
	#f4 : canal na linha 3
	
	mChosenChns = np.zeros((4, len(mEDF[0])))
	
	mChosenChns[0,:] = mEDF[0,:]
	mChosenChns[1,:] = mEDF[1,:]
	mChosenChns[2,:] = mEDF[2,:]
	mChosenChns[3,:] = mEDF[3,:]
	
	return mChosenChns
	
	
# Metodo para leitura de arquivos .csv
def readCSV(filename):
	#pula linha de cabeçalho
	my_data = np.genfromtxt(filename, delimiter=',',skip_header=1)
	return my_data

# Retorna um array com os limiares para cada arquivo CSV passado como parâmetro
def thresholdValenceValues(csvFile):
	thresholdLabels = []
	my_data = readCSV(csvFile)
	for i in range(len(my_data)):
		if my_data[i][0] > 5:
			thresholdLabels.append(1)
		else:
			thresholdLabels.append(0)
	return thresholdLabels

def mainMatrixGenerator(edfPathName):
		#Contador para controle dos vídeos
		count = 1
		#Criação da Matriz principal com 20 linhas e 516 colunas
		mainMatrix = np.zeros(shape = (20 , 516)) # essa quantidade de colunas não deve ser fixa!
		#Laço para capturar os sinais dos 20 vídeos de cada parte
		while count <= 20:
			edfFileName = edfPathName + '/Video' + str(count) + '.edf'
			mChosenChns = getChosenChannels(edfFileName)
			
			psd = computePSD(mChosenChns)
			
			#pdb.set_trace()
			#Adiciona uma nova linha de vídeo a matriz principal 
			mainMatrix[count - 1] = psd
			count += 1
		return mainMatrix

if __name__ == "__main__" :
	#print 'Hello!'

	labels1FileName = sys.argv[1]
	edfPathName1 = sys.argv[2]
	
	labels2FileName = sys.argv[3]
	edfPathName2 = sys.argv[4]
	
	labels3FileName = sys.argv[5]
	edfPathName3 = sys.argv[6]	

	#labels1 = readCSV(labels1FileName)
	thresholds1 = thresholdValenceValues(labels1FileName)	
	mainMatrix1 = mainMatrixGenerator(edfPathName1)
	
	thresholds2 = thresholdValenceValues(labels2FileName)	
	mainMatrix2 = mainMatrixGenerator(edfPathName2)
	
	thresholds3 = thresholdValenceValues(labels3FileName)	
	mainMatrix3 = mainMatrixGenerator(edfPathName3)
	
	allLabels = np.concatenate((thresholds1, thresholds2, thresholds3))
	bigMat = np.concatenate((mainMatrix1, mainMatrix2, mainMatrix3))
	
	labelsPlusData = np.zeros((len(allLabels), len(bigMat[0])+1))
	
	labelsPlusData[:,0] = allLabels[:]
	
	labelsPlusData[:,1:] = bigMat
	
	np.random.shuffle(labelsPlusData)

	#pdb.set_trace()

	scaler = preprocessing.StandardScaler().fit(labelsPlusData[:,1:])
	
	mTrn = labelsPlusData[0:40,:]
	mTst = labelsPlusData[40:,:]
	
	XTrn = scaler.transform(mTrn[:,1:])
	XTst = scaler.transform(mTst[:,1:])
	YTst = mTst[:,0]
	
	clf = svm.SVC()
	clf.fit(XTrn, mTrn[:,0])  
	#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    	Ypredicted = clf.predict(XTst)
    		
	acc = accuracy_score(YTst, Ypredicted)
	
	f1 = f1_score(YTst, Ypredicted, average='micro')
	
	rc = recall_score(YTst, Ypredicted, average='micro')
	
	pr = precision_score(YTst, Ypredicted, average='micro')
	
	print 'accuracy: '+str(acc)
	print 'f1 score: '+str(f1)
	print 'precision: '+str(pr)
	print 'recall: '+str(rc)
	
	#pdb.set_trace()
	
	#print len(mChosenChns[0])
	#print psd.shape
	#print psd2.shape
	#print psd3.shape
	#print psd4.shape
	#pdb.set_trace()
	# próximo passo: realizar o treinamento do classificador SVM: http://scikit-learn.org/stable/modules/svm.html
