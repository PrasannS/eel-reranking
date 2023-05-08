# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
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
r"""
XLM-RoBERTa Encoder
==============
    Pretrained XLM-RoBERTa  encoder from Hugging Face.
"""
from typing import Dict

import torch
from tfr_models.encoders.base import Encoder
from tfr_models.encoders.bert import BERTEncoder
from transformers import XLMRobertaModel, XLMRobertaTokenizer, RobertaModel, RobertaTokenizer, BartModel
import torch
from collections import namedtuple
from transformers import BartModel, AutoTokenizer
from .custom_modeling_bart import CustomBartModel

class BartEncoder(BERTEncoder):
    """RoBERTA Encoder encoder.

    :param pretrained_model: Pretrained model from hugging face.
    """

    def __init__(self, pretrained_model: str) -> None:
        super(Encoder, self).__init__()
        # add special tokens for compatibility with original model thing
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        new_tokens = ['<H>', '<R>', '<T>']
        new_tokens_vocab = {}
        new_tokens_vocab['additional_special_tokens'] = []
        for idx, t in enumerate(new_tokens):
            new_tokens_vocab['additional_special_tokens'].append(t)
        num_added_toks = self.tokenizer.add_special_tokens(new_tokens_vocab)
        # first get cond gen model
        ckpt = torch.load("/mnt/data1/prasann/latticegen/lattice-generation/parent_explore/plms-graph2text/webnlg-bart-base.ckpt")
        state_dict = ckpt['state_dict']
        # make weight keys compatible 
        for key in list(state_dict.keys()):
            if key[:len("model.")]=="model.":
                state_dict[key[len("model."):]] = state_dict.pop(key)
        # TODO let's check and see if this works correctly
        self.model = CustomBartModel.from_pretrained(
            "facebook/bart-base", state_dict=ckpt['state_dict'], vocab_size=50268
        )
        print("using correct weight adjustments")
        self.model.encoder.output_hidden_states = True

    @classmethod
    def from_pretrained(cls, pretrained_model: str) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.
        :param pretrained_model: Name of the pretrain model to be loaded.

        :return: Encoder model
        """
        return BartEncoder(pretrained_model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, decoder_input_ids: torch.Tensor, decoder_attention_mask: torch.Tensor,  **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        if "position_ids" in kwargs and kwargs['position_ids'] is not None:
            modelout = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                custom_position_ids=kwargs['position_ids'],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            modelout = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        
        return {
            "sentemb": modelout.encoder_last_hidden_state[:, 0, :],
            "wordemb": modelout.encoder_last_hidden_state,
            "all_layers": modelout.encoder_hidden_states,
            "attention_mask": modelout.encoder_attentions,
            "hypwordemb": modelout.last_hidden_state,
        }

class RobertaEncoder(BERTEncoder):
    """RoBERTA Encoder encoder.

    :param pretrained_model: Pretrained model from hugging face.
    """

    def __init__(self, pretrained_model: str) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
        if False:
            self.model = RobertaModel.from_pretrained(
                pretrained_model, add_pooling_layer=False, position_embedding_type='relative_key_query'
            )
            print("using relative key query embedding")
        else:
            self.model = RobertaModel.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
       
        self.model.encoder.output_hidden_states = True

    @classmethod
    def from_pretrained(cls, pretrained_model: str) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.
        :param pretrained_model: Name of the pretrain model to be loaded.

        :return: Encoder model
        """
        return RobertaEncoder(pretrained_model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        if "position_ids" in kwargs and kwargs['position_ids'] is not None:
            last_hidden_states, _, all_layers = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=kwargs['position_ids'],
                output_hidden_states=True,
                return_dict=False,
            )
        else:
            last_hidden_states, _, all_layers = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=False,
            )
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }

class XLMREncoder(BERTEncoder):
    """XLM-RoBERTA Encoder encoder.

    :param pretrained_model: Pretrained model from hugging face.
    """

    def __init__(self, pretrained_model: str) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model)
        if False:
            self.model = XLMRobertaModel.from_pretrained(
                pretrained_model, add_pooling_layer=False, position_embedding_type='relative_key_query'
            )
            print("using relative key query embedding")
        else:
            self.model = XLMRobertaModel.from_pretrained(
                pretrained_model, add_pooling_layer=False
            )
       
        self.model.encoder.output_hidden_states = True

    @classmethod
    def from_pretrained(cls, pretrained_model: str) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.
        :param pretrained_model: Name of the pretrain model to be loaded.

        :return: Encoder model
        """
        return XLMREncoder(pretrained_model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        if "position_ids" in kwargs and kwargs['position_ids'] is not None:
            last_hidden_states, _, all_layers = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=kwargs['position_ids'],
                output_hidden_states=True,
                return_dict=False,
            )
        else:
            last_hidden_states, _, all_layers = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=False,
            )
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }

class XLMRDecoder(BERTEncoder):
    """XLM-RoBERTA Encoder encoder.

    :param pretrained_model: Pretrained model from hugging face.
    """

    def __init__(self, pretrained_model: str) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model)
        self.model = XLMRobertaModel.from_pretrained(
            pretrained_model, add_pooling_layer=False, is_decoder=True, 
            add_cross_attention=True
        )
        self.model.encoder.output_hidden_states = True

    @classmethod
    def from_pretrained(cls, pretrained_model: str) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.
        :param pretrained_model: Name of the pretrain model to be loaded.

        :return: Encoder model
        """
        return XLMRDecoder(pretrained_model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, enc_hid_states: torch.Tensor, 
        enc_amask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        last_hidden_states, _, all_layers, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc_hid_states,
            encoder_attention_mask=enc_amask,
            output_hidden_states=True,
            return_dict=False,
        )
        #print(enc_hid_states.shape)
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }
