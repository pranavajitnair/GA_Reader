import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class GRU(nn.Module):
        def __init__(self,input_size,hidden_size):
                super(GRU,self).__init__()
                
                self.gru=nn.GRU(input_size,hidden_size,bidirectional=True,batch_first=True)
                
        def forward(self,input,input_mask):
                seq_len=torch.sum(input_mask,dim=-1)
                sorted_len,sorted_index=seq_len.sort(0,descending=True)
                i_sorted_index=sorted_index.view(-1,1,1).expand_as(input)
                sorted_input=input.gather(0,i_sorted_index.long())
                
                packed_seq=pack_padded_sequence(sorted_input,sorted_len,batch_first=True)
                output,(hidden,cell_state)=self.gru(packed_seq)
                unpacked_seq,unpacked_len=pad_packed_sequence(output,batch_first=True)
                
                _,original_index=sorted_index.sort(0,descending=False)
                unsorted_index=original_index.view(-1,1,1).expand_as(unpacked_seq)
                output_final=unpacked_seq.gather(0,unsorted_index.long())
                
                return output_final,seq_len


class Char_Embeds(nn.Module):
        def __init__(self,n_chars,char_size,embed_size,hidden_size):
                super(Char_Embeds,self).__init__()
                self.hidden_size=hidden_size
                self.embed_size=embed_size
                
                self.char_embedding=nn.Embedding(n_chars,char_size)
                self.forward_project=nn.Linear(hidden_size,embed_size//2)
                self.backward_project=nn.Linear(hidden_size,embed_size//2)
                
                self.gru=GRU(char_size,hidden_size)
                
        def forward(self,input,mask,doc_char,query_char):
                input=self.char_embedding(input)
                input,seq_len=self.gru(input,mask)
                
                final_index=(seq_len-1).view(-1,1).expand(input.size(0),input.size(2)).unsqueeze(1)
                output=input.gather(1,final_index.long()).squeeze()
                
                forward_output=output[:,:self.hidden_size]
                backward_output=output[:,self.hidden_size:]
                forward_output=self.forward_project(forward_output)
                backward_output=self.backward_project(backward_output)
                final=forward_output+backward_output
                
                doc_embed=final.index_select(0,doc_char.view(-1)).view(doc_char.shape[0],
                                                                       doc_char.shape[1],self.embed_size//2)
                query_embed=final.index_select(0,query_char.view(-1)).view(query_char.shape[0],
                                                                           query_char.shape[1],self.embed_size//2)
                
                return doc_embed,query_embed
            
            
class GA_Reader(nn.Module):
        def __init__(self,n_chars,char_size,embed_size,hidden_size_char,hidden_size,
                     vocab_size,pretrained_weights,gru_layers,use_features,use_chars):
                super(GA_Reader,self).__init__()
                self.embedding=nn.Embedding.from_pretrained(pretrained_weights)
                self.use_chars=use_chars
                self.use_features=use_features
                self.gru_layers=gru_layers
                
                self.grus_docs=nn.ModuleList()
                self.grus_query=nn.ModuleList()
                for i in range(gru_layers-1):
                        if i==0:
                                if self.use_chars:
                                        G1=GRU(3*embed_size//2,hidden_size)
                                else:
                                        G1=GRU(embed_size,hidden_size)
                        else:
                                G1=GRU(2*hidden_size,hidden_size)
                        if self.use_chars:
                                G2=GRU(3*embed_size//2,hidden_size)
                        else:
                                G2=GRU(embed_size,hidden_size)
                        self.grus_docs.append(G1)
                        self.grus_query.append(G2)
                        
                if use_features:
                        self.features=nn.Embedding(2,2)
                self.finalgru_doc=GRU(2*hidden_size+use_features*2,hidden_size)
                self.finalgru_query=GRU(3*embed_size//2,hidden_size)
                
                if use_chars:
                        self.char_embeds=Char_Embeds(n_chars,char_size,embed_size,hidden_size_char)
                        
        def forward(self,doc,doc_char,doc_mask,query,query_char,query_mask,
                    char_type,char_type_mask,ans,cloze,cands,cand_mask,qe_comm):
                doc_embed=self.embedding(doc)
                query_embed=self.embedding(query)
                
                if self.use_chars:
                        doc_char_embed,query_char_embed=self.char_embeds(char_type,char_type_mask,doc_char,query_char)
                        doc_embed=torch.cat([doc_embed,doc_char_embed],dim=-1)
                        query_embed=torch.cat([query_embed,query_char_embed],dim=-1)
                        
                for i in range(self.gru_layers-1):
                        doc_D,_=self.grus_docs[i](doc_embed,doc_mask)
                        Q,_=self.grus_query[i](query_embed,query_mask)
                        
                        doc_embed=self.attention(doc_D,Q,doc_mask,query_mask)
                        
                if self.use_features:
                        features=self.features(qe_comm)
                        D=torch.cat([doc_embed,features],dim=-1)
                        
                final_doc,_=self.finalgru_doc(D,doc_mask)
                final_query,_=self.finalgru_query(query_embed,query_mask)
                output=self.attention_sum(final_doc,final_query,cloze,cands,cand_mask)
                        
                return output
                        
        def attention(self,D,Q,doc_mask,query_mask):
                mask_Q=query_mask.unsqueeze(1).expand(-1,D.shape[1],-1)
                mask_D=doc_mask.unsqueeze(-1).expand(-1,-1,Q.shape[1])
                attn_temp=torch.bmm(D,Q.transpose(-1,-2))
                attn_temp=attn_temp+(1-mask_Q)*1e-9+(1-mask_D)*1e-9
                attn=F.softmax(attn_temp,dim=-1)
                
                weights=torch.bmm(attn,Q)
                output=weights*D
                
                return output
            
        def attention_sum(self,doc,query,cloze,cand,cand_mask):
                mask=cloze.view(-1,1).unsqueeze(-1).expand(-1,query.shape[1],query.shape[-1])
                q=query.gather(1,mask)
                q=q[:,0,:].view(query.shape[0],-1,1)
                distribution=torch.bmm(doc,q).squeeze()
                
                probs=F.softmax(distribution,dim=-1)*cand_mask
                output=torch.bmm(probs.unsqueeze(1),cand.float()).squeeze()
                
                return output