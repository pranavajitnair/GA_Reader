import random
import torch


class DataLoader(object):
        def __init__(self,training_data,batch_size):
                self.training_data=training_data
                self.batch_size=batch_size
        
        def get_data(self):
                data=random.choices(self.training_data,k=self.batch_size)
                
                return data
            
        def __load_next__(self):
                data=self.get_data()
                
                max_query_len,max_doc_len,max_cand_len,max_word_len=0,0,0,0
                ans=[]
                clozes=[]
                word_types={}
                for i,instance in enumerate(data):
                        doc,query,doc_char,query_char,cand,ans_,cloze_=instance
                        max_doc_len=max(max_doc_len,len(doc))
                        max_query_len=max(max_query_len,len(query))
                        max_cand_len=max(max_cand_len,len(cand))
                        ans.append(ans_[0])
                        clozes.append(cloze_[0])
                        
                        for index,word in enumerate(doc_char):
                                max_word_len=max(max_word_len,len(word))
                                if tuple(word) not in word_types:
                                        word_types[tuple(word)]=[]
                                        word_types[tuple(word)].append((1,i,index))
                        for index,word in enumerate(query_char):
                                max_word_len=max(max_word_len,len(word))
                                if tuple(word) not in word_types:
                                        word_types[tuple(word)]=[]
                                        word_types[tuple(word)].append((0,i,index))
                                        
                docs=torch.zeros(self.batch_size,max_doc_len,dtype=torch.long)
                queries=torch.zeros(self.batch_size,max_query_len,dtype=torch.long)
                cands=torch.zeros(self.batch_size,max_doc_len,max_cand_len,dtype=torch.long)
                docs_mask=torch.zeros(self.batch_size,max_doc_len,dtype=torch.long)
                queries_mask=torch.zeros(self.batch_size,max_query_len,dtype=torch.long)
                cand_mask=torch.zeros(self.batch_size,max_doc_len,dtype=torch.long)
                qe_comm=torch.zeros(self.batch_size,max_doc_len,dtype=torch.long)
                answers=torch.tensor(ans,dtype=torch.long)
                clozes=torch.tensor(clozes,dtype=torch.long)
                
                for i,instance in enumerate(data):
                        doc,query,doc_char,query_char,cand,ans_,cloze_=instance
                        docs[i,:len(doc)]=torch.tensor(doc)
                        queries[i,:len(query)]=torch.tensor(query)
                        docs_mask[i,:len(doc)]=1
                        queries_mask[i,:len(query)]=1
                        
                        for k,index in enumerate(doc):
                                for j,index_c in enumerate(cand):
                                        if index==index_c:
                                                cands[i][k][j]=1
                                                cand_mask[i][k]=1
                                                
                                for y in query:
                                        if y==index:
                                                qe_comm[i][k]=1
                                                break

                        for x,cl in enumerate(cand):
                                if cl==answers[i]:
                                        answers[i]=x
                                        break
                                                
                doc_char=torch.zeros(self.batch_size,max_doc_len,dtype=torch.long)
                query_char=torch.zeros(self.batch_size,max_query_len,dtype=torch.long)
                char_type=torch.zeros(len(word_types),max_word_len,dtype=torch.long)
                char_type_mask=torch.zeros(len(word_types),max_word_len,dtype=torch.long)
                
                index=0
                for word,word_list in word_types.items():
                        char_type[index,:len(word)]=torch.tensor(list(word))
                        char_type_mask[index,:len(word)]=1
                        for (i,j,k) in word_list:
                                if i==1:
                                        doc_char[j,k]=index
                                else:
                                        query_char[j,k]=index
                        index+=1
                                        
                return docs,doc_char,docs_mask,queries,query_char,queries_mask, \
                    char_type,char_type_mask,answers,clozes,cands,cand_mask,qe_comm
                    
                    
class TestLoader(DataLoader):
        def __init__(self,data,num_examples,batch_size=2):
                self.data=data
                self.examples=num_examples
                self.counter=0
                self.batch_size=batch_size
                
        def reset_counter(self):
                self.counter=0
                
        def get_data(self):
                data=self.data[self.counter:self.count+2]
                self.counter+=2
                if self.counter==self.examples:
                        self.reset_counter()
                        
                return data