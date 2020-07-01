import os
import pickle
from collections import Counter
import torch
import torch.nn.functional as F

def load_filenames(path):
        names=[]
        for filenames in os.walk(path):
                names.append(filenames)

        names=names[0][2]    
        return names
    
def get_vocabulary(path,filenames):
        word_count=Counter()    
    
        for filename in filenames:
                raw=open(path+filename).readlines()
                doc_raw=raw[2].split() 
                query_raw=raw[4].split() 
                cand_raw=map(lambda x:x.strip().split(':')[0].split(), 
                raw[8:])
                        
                for word in doc_raw:
                        word_count[word]+=1
                for word in query_raw:
                        word_count[word]+=1
                for word in cand_raw:
                        word_count[word[0]]+=1
        
        index=4                      
        word_to_int={}
        int_to_word={}
        char_set=set()
        word_to_int['start'],word_to_int['end'],word_to_int['unk'],word_to_int['pad']=1,2,3,0
        int_to_word[1],int_to_word[2],int_to_word[3],int_to_word[0]='@start','@end','@unk','@pad'
        for word in word_count:
                word_to_int[word]=index
                int_to_word[index]=word
                index+=1
                
                for character in word:
                        char_set.add(character)
        char_set=list(char_set)
        
        index=1               
        char_to_int={}
        int_to_char={}
        char_to_int[' ']=0
        int_to_char[0]=' '
        for character in char_set:
                char_to_int[character]=index
                int_to_char[index]=character
                index+=1
                
        return word_to_int,int_to_word,char_to_int,int_to_char
    
def process_one_file(filename,word_to_int,char_to_int,return_chars=True):
        raw=open(filename).readlines()
        doc_raw=raw[2].split() 
        query_raw=raw[4].split() 
        ans_raw=raw[6].strip() 
        cand_raw=map(lambda x:x.strip().split(':')[0].split(), 
                raw[8:])
       
        query_raw.insert(0,'start')
        query_raw.append('end')
        
        doc,cand,query,ans,cloze=[],[],[],[],[]
        for word in doc_raw:
                doc.append(word_to_int[word])
        for i,word in enumerate(query_raw):
                if word=='@placeholder':
                        cloze.append(i)
                query.append(word_to_int[word])       
        ans.append(word_to_int[ans_raw])
        for word in cand_raw:
                cand.append(word_to_int[word[0]])
                
        if return_chars:
                doc_chars=[]
                query_chars=[]
                
                for word in doc_raw:
                        temp=[]
                        for char in word:
                                temp.append(char_to_int[char])
                        doc_chars.append(temp)
                for word in query_raw:
                        temp=[]
                        for char in word:
                                temp.append(char_to_int[char])
                        query_chars.append(temp)
            
        return (doc,query,doc_chars,query_chars,cand,ans,cloze)
    
def process_all_files(path,return_chars=True):
        filenames=load_filenames(path)
        word_to_int,int_to_word,char_to_int,int_to_char=get_vocabulary(path,filenames)
        training_data=[]
        
        for filename in filenames:
                file_path=path+filename
                instance=process_one_file(file_path,word_to_int,char_to_int,return_chars)
                training_data.append(instance)
                
        return word_to_int,int_to_word,char_to_int, \
                int_to_char,training_data
                
def load_GloVe(path,word_to_int,dim=100):
        fp = open(path,encoding='utf-8')
        word_to_vec={}

        for lines in fp:
                vector=lines.split()
                word=vector[0]
                if word in word_to_int:
                         vector=vector[1:]
                         vector=[float(k) for k in vector]
                         word_to_vec[word]=vector

        glove=torch.zeros(len(word_to_int),dim)
        for word,vector in word_to_vec.items():
                glove[word_to_int[word]]=torch.tensor(vector)
                
        return glove
    
def accuracy_cal(output,answer):
        pred=F.softmax(output,dim=-1)
        _,pred=pred.max(dim=-1)
        count=0
        
        for i in range(pred.shape[0]):
                if pred[i]==answer[i]:
                        count+=1
                        
        return count

def store_in_file(word_to_int,int_to_word,char_to_int,
                  int_to_char,training_data):
        file=open('training.pickle','wb')
        store={}
        
        store['word_to_int']=word_to_int
        store['int_to_word']=int_to_word
        store['char_to_int']=char_to_int
        store['int_to_char']=int_to_char
        store['training_data']=training_data
        
        pickle.dump(store,file)

def read_from_file(file_name):
        file=open(file_name,"rb")
        store=pickle.load(file)

        return store['word_to_int'],store['int_to_word'], \
          store['char_to_int'],store['int_to_char'], store['training_data']