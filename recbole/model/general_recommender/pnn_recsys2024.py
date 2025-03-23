# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import EmbLoss
from recbole.utils import InputType
from recbole.model.loss import BPRLoss


class PNN_RecSys2024(GeneralRecommender):
    input_type = InputType.PAIRWISE
    
    @staticmethod
    def Euclidean(x: Tensor, y: Tensor):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()
    @staticmethod
    def uniform_loss(x: Tensor):
        x = F.normalize(x, dim=-1)
        # torch.pdist(, p=2) で, 各 neutral item vector 事の距離を算出しておく. また、対象は　entity (item or user) 次元に沿って平均を取る
        return torch.pdist(x.mean(dim=-2).squeeze(-2), p=2).pow(2).mul(-2).exp().mean().log()

    def __init__(self, config, dataset):
        super(PNN_RecSys2024, self).__init__(config, dataset)

        # ユーザーが過去にインタラクションしたアイテムを取得
        self.history_item_id, _, self.history_item_len = dataset.history_item_matrix(
            max_history_len=config["history_len"]
        )
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_len = self.history_item_len.to(self.device)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.alpha = config["alpha"]
        self.beta = config['beta']
        self.neg_seq_len = config["train_neg_sample_args"]["sample_num"] # TODO: 論文では本来ユーザーの正例の数だけ選ぶようにしている。ただ、一旦は一律でネガティブサンプリング数を増やすことで対応
        self.reg_weight = config["reg_weight"]
        self.history_len = torch.max(self.history_item_len, dim=0)

        # user embedding matrix
        self.user_emb = nn.Embedding(self.n_users, self.embedding_size)
        # item embedding matrix
        self.item_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.W_k = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
            )
        # dropout
        self.lambda_adapter = nn.Linear(self.embedding_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.require_pow = config["require_pow"]
        # l2 embedding parameter regularization loss
        self.reg_loss = EmbLoss()
        self.BPR = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        # get the mask
        self.item_emb.weight.data[0, :] = 0
    


    def user_pos_item_attention(self, user_e: Tensor, history_positive_item_e: Tensor) -> Tensor:

        # [user_num, max_history_len, embedding_size]
        key = self.W_k(history_positive_item_e)
        beta_attention = torch.matmul(key, user_e.unsqueeze(2)).squeeze(2) # β_{i}^{attr}: 各 positive item に対する attention
        alpha_attention = torch.exp(beta_attention)
        mask = (history_positive_item_e.sum(dim=-1) != 0).int()
        alpha_attention = alpha_attention * mask
        # [user_num, max_history_len]
        alpha_attention_weight = alpha_attention / (
            alpha_attention.sum(dim=1, keepdim=True) + 1.0e-10
        )
        # [user_num, embedding_size]
        out = torch.matmul(alpha_attention_weight.unsqueeze(1), history_positive_item_e).squeeze(1) # 式(10)の σ() の中身. attention weight をかけて、ベクトルを重み付け
        return out

    def score(self, user_e, item_e):
        score = torch.sum(torch.mul(user_e, item_e), axis=-1)
        return score

    def forward(self, user, pos_item, history_item, history_len, neg_item_seq):

        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, embedding_size]
        pos_item_e = self.item_emb(pos_item)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)
        # [nuser_num, neg_seq_len, embedding_size]
        neg_item_seq_e = self.item_emb(neg_item_seq)

        pos_cos = self.score(user_e, pos_item_e)
        neg_cos = self.score(user_e.unsqueeze(dim=1), neg_item_seq_e)


        #semi-supervised learning
        
        max_indices = torch.max(neg_cos, dim=1)[1].detach()  # [batch_size] , 評価されていない中で、ユーザーとのコサイン類似度が最大のものを選ぶ
        user_pos_item_attention = self.user_pos_item_attention(user_e, history_item_e)
        lambdat = torch.sigmoid(self.lambda_adapter(user_pos_item_attention).mean()) # Eq(10). 実際はミニバッチになっているので、各 [user, history_item] 事に算出される lambda の平均を取る
        
        # neutral item の選択
        hard_item = torch.gather(neg_item_seq, dim=1, index=max_indices.unsqueeze(-1)).squeeze() # neutral item の選択
        hard_neg_e = self.item_emb(hard_item)
        hard_neg_cos = self.score(user_e, hard_neg_e)
        bpr_loss = self.BPR(pos_cos, hard_neg_cos)
        
        # classifier

        min_indices = torch.min(neg_cos, dim=1)[1].detach()  # [batch_size]
        neg_item = torch.gather(neg_item_seq, dim=1, index=min_indices.unsqueeze(-1)).squeeze()
        true_neg_e = self.item_emb(neg_item)
        true_neg_cos = self.score(user_e, true_neg_e)

        pos_direction = torch.sign(pos_item_e.unsqueeze(dim=1))
        neg_direction = torch.sign(true_neg_e.unsqueeze(dim=1))

        pos_random_noise = torch.rand(pos_direction.shape).to(self.device)
        neg_random_noise = torch.rand(neg_direction.shape).to(self.device) 

        pos_random_noise = torch.nn.functional.normalize(pos_random_noise, p=2, dim=-1) * 0.1        
        neg_random_noise = torch.nn.functional.normalize(neg_random_noise, p=2, dim=-1) * 0.1

        e_pos_clamp = neg_item_seq_e + pos_random_noise
        e_neg_clamp = neg_item_seq_e + neg_random_noise



        loss_constrain = self.Euclidean(e_pos_clamp , e_neg_clamp)
        loss_uniform = self.uniform_loss(neg_item_seq_e)

        loss_rank = self.BPR(pos_cos,neg_cos.mean(dim=-1)) + self.BPR(neg_cos.mean(dim=-1),true_neg_cos)


        
        final_loss = lambdat * bpr_loss  + (1-lambdat)*(loss_rank + self.alpha * loss_constrain + self.beta * loss_uniform)

        # l2 regularization loss
        reg_loss = self.reg_loss(
            user_e,
            pos_item_e,
            # history_item_e,
            neg_item_seq_e,
            require_pow=self.require_pow,
        )

        loss = final_loss + self.reg_weight * reg_loss.sum()
        return loss

    def calculate_loss(self, interaction):

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # get your neg items based on your sampling strategy
        # TODO: 論文では、user 事の interaction 履歴の長さによって、サンプリング数を変更している。が、現状できていない...
        neg_item_seq = neg_item.reshape((self.neg_seq_len, -1))
        neg_item_seq = neg_item_seq.T
        
        user_number = int(len(user) / self.neg_seq_len)
        # user's id
        user = user[0:user_number]
        # インタラクションしたアイテム
        history_item = self.history_item_id[user]
        # positive item's id
        pos_item = pos_item[0:user_number]
        # インタラクションしたアイテムの数
        history_len = self.history_item_len[user]
        loss = self.forward(user, pos_item, history_item, history_len, neg_item_seq)
        return loss



    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        # user_e, item_e = self.forward(user, item)
        user_e = self.user_emb(user)
        # [user_num, embedding_size]
        item_e = self.item_emb(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_emb(user)
        all_item_e = self.item_emb.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)