from transformers import AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .attention import MultiHeadAttention

import os

CLASSES = {"NEI":1, "SUPPORTED":0, "REFUTED":2}

class Model(nn.Module):
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __init__(self, args=None, weight_label = None):
        super(Model, self).__init__()
        self.model_dir = args.model_dir
        self.cuda_flag = args.cuda
        self.clip_grad_norm = args.clip_grad_norm
        self.cross_attention = args.cross_attention
        self.sentence_embedding_model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
        for param in self.sentence_embedding_model.parameters():
            param.requires_grad = False
        self.attention_verdict = MultiHeadAttention(args.embedding_size, args.num_head)
        self.classify_verdict = nn.Linear(args.embedding_size, args.verdict_size)
        
        torch.nn.init.xavier_uniform(self.classify_verdict.weight)
        self.dropout = nn.Dropout(p = args.dropout_rate)


        self.optimizer = optim.AdamW(self.parameters(), lr = args.lr_rate)
        if weight_label is not None:
            self.loss = nn.CrossEntropyLoss(weight=torch.tensor(weight_label))
        else:
            self.loss = nn.CrossEntropyLoss()

        self.global_step = 0



    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.global_step += 1
        return loss

    def forward(self, batch_input):
        # reshape (bacth, sentence, sequence) to (batch * sentence, sequence)
        context_shape = batch_input["context"]["input_ids"].size()
        batch_input["context"]["input_ids"] = batch_input["context"]["input_ids"].view(-1, 256)
        batch_input["context"]["attention_mask"] = batch_input["context"]["attention_mask"].view(-1, 256)
        batch_input["context"]["token_type_ids"] = batch_input["context"]["token_type_ids"].view(-1, 256)

        batch_input["claim"]["input_ids"] = batch_input["claim"]["input_ids"].view(-1, 256)
        batch_input["claim"]["token_type_ids"] = batch_input["claim"]["token_type_ids"].view(-1, 256)
        batch_input["claim"]["attention_mask"] = batch_input["claim"]["attention_mask"].view(-1, 256)
        
        if self.cuda_flag:
            batch_input["context"]["input_ids"] = batch_input["context"]["input_ids"].cuda()
            batch_input["context"]["attention_mask"] = batch_input["context"]["attention_mask"].cuda()
            batch_input["context"]["token_type_ids"] = batch_input["context"]["token_type_ids"].cuda()

            batch_input["claim"]["input_ids"] = batch_input["claim"]["input_ids"].cuda()
            batch_input["claim"]["token_type_ids"] = batch_input["claim"]["token_type_ids"].cuda()
            batch_input["claim"]["attention_mask"] = batch_input["claim"]["attention_mask"].cuda()
            batch_input["context_sentence_mask"] = batch_input["context_sentence_mask"].cuda()
            batch_input["verdict"] = batch_input["verdict"].cuda()
            batch_input["evidence_index"] = batch_input["evidence_index"].cuda()
        #embed sentences
        with torch.no_grad():
            model_output_context = self.sentence_embedding_model(**batch_input["context"])
            model_output_claim =  self.sentence_embedding_model(**batch_input["claim"])

        sentence_embeddings_context = self.mean_pooling(model_output_context, batch_input["context"]['attention_mask'])
        sentence_embeddings_claim = self.mean_pooling(model_output_claim, batch_input["claim"]['attention_mask'])

        #reshape (batch * sentence, sequence) to (bacth, sentence, sequence)
        sentence_embeddings_context = sentence_embeddings_context.view(context_shape[0], context_shape[1], -1)

        if self.cross_attention:
#             attention_mask = batch_input["context_sentence_mask"] @ batch_input["context_sentence_mask"].T
#             print(attention_mask.size())
            batch_input["context_sentence_mask"] = batch_input["context_sentence_mask"].unsqueeze(1)
            
            attention_vector = self.attention_verdict(sentence_embeddings_claim.unsqueeze(1), sentence_embeddings_context, sentence_embeddings_context, mask=batch_input["context_sentence_mask"])
        else:
            attention_logits = torch.bmm(sentence_embeddings_context, sentence_embeddings_claim.unsqueeze(2))
            attention_logits = attention_logits.squeeze(-1)
            attention_logits = attention_logits - (1 - batch_input["context_sentence_mask"]) * 1e9
            attention_weights = nn.Softmax(dim=-1)(attention_logits)
            attention_weights = self.dropout(attention_weights)
            attention_vector = torch.bmm(torch.transpose(sentence_embeddings_context, 1, 2), attention_weights.unsqueeze(2))
            attention_vector = attention_vector.squeeze(-1)

        class_logits = self.classify_verdict(attention_vector)
        class_logits = class_logits.squeeze(1)
        
        loss = self.loss(class_logits, batch_input["verdict"])
        class_labels = class_logits.argmax(dim=-1)
        verdict_acc = class_labels == batch_input["verdict"]

        sentence_embeddings_claim = sentence_embeddings_claim.unsqueeze(1)
        sim_cos = F.cosine_similarity(sentence_embeddings_claim, sentence_embeddings_context, dim = -1).argmax(dim=-1) + 1

        sim_cos = sim_cos * (batch_input["verdict"] != CLASSES["NEI"])
        evidence_acc = sim_cos == batch_input["evidence_index"]
        # print(loss)
        return evidence_acc, verdict_acc, loss


    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        global_step_padded = format(self.global_step, '08d')
        ckpt_name = 'ckpt-' + global_step_padded
        path = os.path.join(self.model_dir, ckpt_name)
        ckpt = self.state_dict()
        torch.save(ckpt, path)
        
    
    def load_model(self, model_name):
        path = os.path.join(self.model_dir, model_name)
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        