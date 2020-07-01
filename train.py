import torch.nn as nn
import torch.optim as optim
import argparse
import os

from utils import process_all_files,load_GloVe,accuracy_cal
from model import GA_Reader
from data_loader import DataLoader,TestLoader

def train(epochs,iterations,loader_train,loader_val,
          model,optimizer,loss_function):
        for epoch in range(epochs):            
                for iteration in range(iterations):
                        model.train()
                        optimizer.zero_grad()    

                        doc,doc_char,doc_mask,query,query_char,query_mask, \
                            char_type,char_type_mask,answer,cloze,cand, \
                                cand_mask,qe_comm=loader_train.__load_next__()
                                
                        output=model( doc,doc_char,doc_mask,query,query_char,query_mask,
                            char_type,char_type_mask,answer,cloze,cand,
                                cand_mask,qe_comm)
                        
                        loss=loss_function(output,answer)
                        scalar=loss.item()
                        loss.backward()
                        optimizer.step()

                        valid_loss,valid_acc=validate(loader_val,model,loss_function)
                        print('epoch=',epoch+1,'iteration=',iteration+1,'training loss=',scalar,
                              'validation loss=',valid_loss,'validation accuracy=',valid_acc)
                        
                if epoch>=2:
                        optimizer=optim.Adam(model.parameters(),lr=optimizer.param_groups[0]['lr']/2)

def validate(loader_val,model,loss_function):
        model.eval()
        return_loss=0
        accuracy=0
        
        for _ in range(loader_val.examples//loader_val.batch_size):
                doc,doc_char,doc_mask,query,query_char,query_mask, \
                                    char_type,char_type_mask,answer,cloze,cand, \
                                        cand_mask,qe_comm=loader_val.__load_next__()
                                        
                output=model( doc,doc_char,doc_mask,query,query_char,query_mask,
                    char_type,char_type_mask,answer,cloze,cand,
                        cand_mask,qe_comm)
        
                accuracy+=accuracy_cal(output,answer)
                loss=loss_function(output,answer)
                return_loss+=loss.item()
        
        return_loss/=(loader_val.examples//loader_val.batch_size)
        accuracy=100*accuracy/loader_val.examples
        
        return return_loss,accuracy

def test(loader_test,model):
        model.eval()
        accuracy=0
        for _ in range(loader_test.examples//loader_test.batch_size):
                doc,doc_char,doc_mask,query,query_char,query_mask, \
                                    char_type,char_type_mask,answer,cloze,cand, \
                                        cand_mask,qe_comm=loader_test.__load_next__()
                                        
                output=model( doc,doc_char,doc_mask,query,query_char,query_mask,
                    char_type,char_type_mask,answer,cloze,cand,
                        cand_mask,qe_comm)
        
                accuracy+=accuracy_cal(output,answer)
                
        accuracy=100*accuracy/loader_test.examples
        print('test accuracy=',accuracy)
        
def main(args):
        word_to_int,int_to_word,char_to_int,int_to_char, \
            training_data=process_all_files(args.train_file)
        glove_embeddings=load_GloVe(args.embed_file,word_to_int,args.embed_size)
        loss_function=nn.CrossEntropyLoss()
        
        model=GA_Reader(len(char_to_int),args.char_size,args.embed_size,
                        args.char_hidden_size,args.hidden_size,len(word_to_int),
                        glove_embeddings,args.gru_layers,args.use_features,args.use_char)
        
        optimizer=optim.Adam(model.parameters(),lr=args.lr)
        data_loader_train=DataLoader(training_data[:args.training_size],args.batch_size)
        data_loader_validate=TestLoader(training_data[args.training_size:args. \
                                                      training_size+args.dev_size],args.dev_size)
        data_loader_test=TestLoader(training_data[args. \
                                                  training_size_args.dev_size:args. \
                                                      training_size+args.dev_size+args.test_size],args.test_size)
        
        train(args.epochs,args.iterations,data_loader_train,
              data_loader_validate,model,optimizer,loss_function)
        
        test(data_loader_test,model)
        
def setup():
        parser=argparse.ArgumentParser('argument parser')
        parser.add_argument('--lr',type=float,default=0.00005)
        parser.add_argument('--epochs',type=int,default=12)
        parser.add_argument('--iterations',type=int,default=120)
        parser.add_argument('--hidden_size',type=int,default=256)
        parser.add_argument('--char_hidden_size',type=int,default=50)
        parser.add_argument('--char_size',type=int,default=25)
        parser.add_argument('--embed_size',type=int,default=100)
        parser.add_argument('--use_features',type=bool,default=True)
        parser.add_argument('--use_char',type=bool,default=True)
        parser.add_argument('--batch_size',type=int,default=32)
        parser.add_argument('--gru_layers',type=int,default=3)
        parser.add_argument('--embed_file',type=str,default=os.getcwd()+'/word2vec_glove.text')
        parser.add_argument('--train_file',type=str,default=os.getcwd()+'/train/')
        parser.add_argument('--train_size',type=int,default=380298)
        parser.add_argument('--dev_size',type=int,default=3924)
        parser.add_argument('--test_size',type=int,default=3198)
        
        args=parser.parse_args()
        
        return args
    
if __name__=='__main__':
        args=setup()
        main(args)