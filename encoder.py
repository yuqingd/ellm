import torch.nn as nn 
import torch
from transformers import AutoModel
import torch.nn.functional as F
import os
import pathlib
import pickle as pkl
import time
import numpy as np
import fasteners

class SbertEncoder(nn.Module):
    def __init__(self, embedding_dim, device, debug=True):
        super().__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L3-v2')
        self.model.eval()
        
        self.output_head = nn.Sequential(
            nn.Linear(384, embedding_dim) # for minilm
        )
        # Logging
        self.overall_time = 0
        self.cache_load_time = 0
        self.api_queries = self.cache_queries = 0
        self.cache_lookup_time = self.query_time = self.cache_save_time = 0
        self.linear_time = 0
        self.unit_cache_time_list = [0]
        self.unit_query_time_list = [0]
        self.debug = debug

        self.cache_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / 'dqn_cache.pkl'
        if not self.debug:
            self.rw_lock = fasteners.InterProcessReaderWriterLock(self.cache_path)
        self.cache = {}
        self.device = device
        self.model.eval()
        self.num_cache_load_errors = 0
        

    def log(self):
        metrics = {
            'dqn_overall_time': self.overall_time,
            'dqn_cache_load_time': self.cache_load_time,
            'dqn_api_queries': self.api_queries,
            'dqn_cache_queries': self.cache_queries,
            'dqn_cache_lookup_time': self.cache_lookup_time,
            'dqn_query_time': self.query_time,
            'dqn_cache_save_time': self.cache_save_time,
            'dqn_unit_cache_time': np.mean(self.unit_cache_time_list),
            'dqn_unit_query_time': np.mean(self.unit_query_time_list),
            'dqn_linear_time': self.linear_time,
            'dqn_cache_size': len(self.cache),
            'dqn_cache_load_errors': self.num_cache_load_errors,
        }
        return metrics
        
    def load_cache(self):
        if self.debug:
            return {}
        start_time = time.time()
        if not self.cache_path.exists():
            cache = {}
            with open(self.cache_path, 'wb') as f:
                pkl.dump({}, f)
        else:
            try:
                self.rw_lock.acquire_read_lock()
                with open(self.cache_path, 'rb') as f:
                    cache = pkl.load(f)
                self.rw_lock.release_read_lock()
            except FileNotFoundError:
                self.num_cache_load_errors += 1
                cache = {}
        self.cache_load_time += time.time() - start_time
        return cache
        
    def save_cache(self):
        if self.debug: return
        start_time = time.time()
        self.rw_lock.acquire_write_lock()
        with open(self.cache_path, 'wb') as f:
            pkl.dump(self.cache, f)
        self.rw_lock.release_write_lock()
        self.cache_load_time += time.time() - start_time

    def load_and_save_cache(self):
        new_cache = self.load_cache()
        self.cache = {**new_cache, **self.cache}
        self.save_cache()
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input):
        input = input.long()
        self.model.eval()
        all_start_time = start_time = time.time()
        # Split inputs into those in cache and those not in cache
        in_cache, not_in_cache, not_in_cache_tups, ids_cache, ids_not_cache = [], [], [], [], []
        tuples = [tuple(num for num in i.tolist() if not num == 0) for i in input]
        for index, (tup, i) in enumerate(zip(tuples, input)):
            exists = tup in self.cache
            if exists:
                in_cache.append(tup)
                ids_cache.append(index)
            else:
                not_in_cache.append(i)
                ids_not_cache.append(index)
                not_in_cache_tups.append(tup)
        self.cache_lookup_time += time.time() - start_time
        # Load cache for those in cache
        if len(in_cache) > 0:
            start_time = time.time()
            in_cache_embeddings = torch.stack([self.cache[tup] for tup in in_cache])
            self.cache_lookup_time += time.time() - start_time
            self.cache_queries += len(in_cache)
            self.unit_cache_time_list.append((time.time() - start_time) / len(in_cache))
            self.unit_cache_time_list = self.unit_cache_time_list[-100:]
        else:
            in_cache_embeddings = torch.FloatTensor([]).to(self.device)
        # Query model for those not in cache
        if len(not_in_cache) > 0:
            start_time = time.time()
            not_in_cache_input = torch.stack(not_in_cache)
            not_in_cache_embeddings = self.embed(not_in_cache_input)
            self.query_time += time.time() - start_time
            self.unit_query_time_list.append((time.time() - start_time) / len(not_in_cache))
            self.unit_query_time_list = self.unit_query_time_list[-100:]
            # Save new embeddings in the cache
            ss = time.time()
            for tup, embedding in zip(not_in_cache_tups, not_in_cache_embeddings):
                self.cache[tup] = embedding
            self.cache_save_time += time.time() - ss
            self.api_queries += len(not_in_cache)
        else:
            not_in_cache_embeddings = torch.FloatTensor([]).to(self.device)
        start = time.time()
        # Combine outputs
        index_ids = ids_cache + ids_not_cache
        restore_ids = torch.argsort(torch.IntTensor(index_ids)).tolist()
        embeddings = torch.cat([in_cache_embeddings, not_in_cache_embeddings])
        embeddings = embeddings[restore_ids]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        out = self.output_head(embeddings)
        self.linear_time += time.time() - start
        self.overall_time += time.time() - all_start_time
        return out

    def embed(self, input):
        mask = torch.zeros_like(input).to(self.device) 
        mask[input!=0] = 1
        with torch.no_grad():
            embeddings = self.model(input_ids=input, attention_mask=mask)    
        embeddings = self.mean_pooling(embeddings, mask)
        return embeddings
