# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import transformers.models.roberta.modeling_roberta as roberta


class RobertaForSequenceClassificationEntityMax(roberta.RobertaPreTrainedModel):
    def __init__(self, config, args=None, num_labels=1, entity_drop=0.1):
        super(RobertaForSequenceClassificationEntityMax, self).__init__(config)
        self.roberta = roberta.RobertaModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_entity = nn.Dropout(entity_drop)
        self.classifier = nn.Linear(config.hidden_size * 3, num_labels * 36)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        labels=None,
        n_class=1,
        b_mask=None,
        c_mask=None,
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        enc_mem = outputs[0]                          # [bsz, seq_len, hid_size]
        pooled_output = outputs[1]

        # print('enc_mem', enc_mem.size())
        # print('b_mask', b_mask.size())              # [bsz, seq_len]
        # print('c_mask', c_mask.size())              # [bsz, seq_len]
        # print('pooled_out', pooled_output.size())   # [bsz, hid_size]
        # exit()

        b_mask = b_mask.unsqueeze(-1).repeat(1, 1, enc_mem.size(-1))
        c_mask = c_mask.unsqueeze(-1).repeat(1, 1, enc_mem.size(-1))
        b_pooled = self.dropout_entity(torch.max(enc_mem * b_mask, dim=1).values)       # [bsz, hid_size]
        c_pooled = self.dropout_entity(torch.max(enc_mem * c_mask, dim=1).values)
        # print('b_pooled', b_pooled.size())
        # print('c_pooled', c_pooled.size())
        # exit()

        pooled_output = self.dropout(pooled_output)
        pooled_output_catted = torch.cat(
            [pooled_output, b_pooled, c_pooled], dim=-1
        )  # [bsz, hid_size * 3]
        logits = self.classifier(pooled_output_catted)
        logits = logits.view(-1, 36)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, 36)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
