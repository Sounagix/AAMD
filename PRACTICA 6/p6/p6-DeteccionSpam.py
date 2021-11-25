import process_email as pmail
import get_vocab_dict as getvoc 
import codecs
import collections
import numpy as np
from sklearn import svm

vocabDict = collections.OrderedDict(getvoc.getVocabDict())
X_input = np.zeros(shape=(1000, len(vocabDict))) 
Y = np.zeros(1000)

for i in range(1,500):  
    Y[i] = 1  
    path = "spam/{:04d}.txt".format(i) 
    email = pmail.email2TokenList(codecs.open (path, 'r', encoding = 'utf-8', errors = 'ignore').read())
    for j in range (len(email)):
        if(email [j] in vocabDict):    
            X_input[i][list(vocabDict.keys()).index(email[j])] = 1
            
spam_processor = svm.SVC(C = 1.0, kernel = 'linear')
spam_processor.fit(X_input, Y.ravel())