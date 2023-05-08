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
ReferencelessRegression
========================
    Referenceless Regression Metric that learns to predict a quality assessment by
    looking at source and translation.
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from tfr_models.models.regression.regression_metric import RegressionMetric
from tfr_models.modules import FeedForward


# make causal masks for a given example
# TODO is there a more efficient approach to this
def causalmask (a, dev):
    masksdef = torch.zeros((a.shape[0], a.shape[1],a.shape[1]), device=dev)
    for i in range(len(a)):
        lim = int(torch.sum(a[i]))
        masksdef[i, :lim, :lim] = torch.tril(torch.ones((lim, lim)))
    return masksdef

def whole_input(mt_mask, src_mask, mt_inp, src_inp, dev):
    with torch.no_grad():
        padds = torch.sum(src_mask, 1).int()
        madds = torch.sum(mt_mask, 1).int() + padds
        madds[madds>512] = 512
        # TODO faster way to do this?
        wholen = min(max(madds), 512)
        ninps = torch.ones((src_inp.shape[0], wholen), device=dev).int()
        # start off mask as all zeros
        mskfull = torch.zeros((src_inp.shape[0], wholen, wholen), device=dev).bool()
        #print(madds)
        #print(wholen)
        for i in range(len(src_inp)):
            # make new inputs
            ninps[i][:padds[i]] = src_inp[i][:padds[i]]
            ninps[i][padds[i]:madds[i]] = mt_inp[i][:madds[i]-padds[i]]
            # build the causal mask, left of mt end is just 1s
            mskfull[i][:madds[i], :padds[i]] = 1
            # bottom right becomes tril, but leave off padding
            mskfull[i][padds[i]:madds[i], padds[i]:madds[i]] = \
                torch.tril(torch.ones_like(mskfull[i][padds[i]:madds[i], padds[i]:madds[i]]))
    return madds.unsqueeze(-1), ninps, mskfull
        

# TODO this can just be the same as reflesseval I think? 
class ReferencelessRegression(RegressionMetric):
    """ReferencelessRegression:

    :param nr_frozen_epochs: Number of epochs (% of epoch) that the encoder is frozen.
    :param keep_embeddings_frozen: Keeps the encoder frozen during training.
    :param optimizer: Optimizer used during training.
    :param encoder_learning_rate: Learning rate used to fine-tune the encoder model.
    :param learning_rate: Learning rate used to fine-tune the top layers.
    :param layerwise_decay: Learning rate % decay from top-to-bottom encoder layers.
    :param encoder_model: Encoder model to be used.
    :param pretrained_model: Pretrained model from Hugging Face.
    :param pool: Pooling strategy to derive a sentence embedding ['cls', 'max', 'avg'].
    :param layer: Encoder layer to be used ('mix' for pooling info from all layers.)
    :param dropout: Dropout used in the top-layers.
    :param batch_size: Batch size used during training.
    :param train_data: Path to a csv file containing the training data.
    :param validation_data: Path to a csv file containing the validation data.
    :param hidden_sizes: Hidden sizes for the Feed Forward regression.
    :param activations: Feed Forward activation function.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "RoBERTa",
        pretrained_model: str = "osunlp/ReasonBERT-RoBERTa-base",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[str] = None,
        validation_data: Optional[str] = None,
        hidden_sizes: List[int] = [1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        load_weights_from_checkpoint: Optional[str] = None,
    ) -> None:
        super(RegressionMetric, self).__init__(
            nr_frozen_epochs,
            keep_embeddings_frozen,
            optimizer,
            encoder_learning_rate,
            learning_rate,
            layerwise_decay,
            encoder_model,
            pretrained_model,
            pool,
            layer,
            dropout,
            batch_size,
            train_data,
            validation_data,
            load_weights_from_checkpoint,
            "referenceless_regression_metric"
        )
        self.save_hyperparameters()

    
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units * 4,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )
        self.encodermod = encoder_model
        """
        self.regressor = nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            # TODO expand this if we need to cat
            nn.Linear(self.encoder.model.config.hidden_size, 1), 
        )
        """
        
    def is_referenceless(self) -> bool:
        return True

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        src_inputs = self.encoder.prepare_sample(sample["src"])
        mt_inputs = self.encoder.prepare_sample(sample["mt"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        inputs = {**src_inputs, **mt_inputs}

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def forward(
        self,
        src_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        mt_input_ids: torch.tensor,
        mt_attention_mask: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        # TODO adapt bart-style model to reflesseval as well
        if self.encoder_model=="BART":
            # TODO need to update later for our reflesseval, posid stuff
            src_sentemb, modelout = self.get_sentence_embedding(src_input_ids, src_attention_mask, mt_input_ids, mt_attention_mask)
            hyp_wordemb = modelout['hypwordemb']
        else:
            # TODO do we need to disable causal mask for sanity?
            src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        
            # encode hypothesis passed in, will use special pos_ids and attention mask passed in 
            hyp_wordemb = self.encoder(mt_input_ids, mt_attention_mask)['wordemb']
        # for inference time, if we need to pass in a custom mask
        # if mask_causal:
        #     hyp_wordemb = self.encoder(mt_input_ids, mt_attention_mask)['wordemb']
        #else:
        #print("!")
        # TODO switch back to causal
        # hyp_wordemb = self.encoder(mt_input_ids, causalmask(mt_attention_mask, dev=self.device))['wordemb']
        #hyp_wordemb = self.encoder(mt_input_ids, mt_attention_mask)['wordemb']

        src_wordemb = src_sentemb[:, None, :].expand(hyp_wordemb.shape)
        sub_embed = hyp_wordemb - src_wordemb
        mult_embed = hyp_wordemb * src_wordemb

        # repr(x_1,...,x_n, y_i) = [y_i, pool(x_1,...,x_n), y_i - pool(x_1,...,x_n), y_i * pool(x_1,...,x_n)]
        # need to do that over i different output toks, is there an easy way to do that?
        embedded_sequences = torch.cat(
            (hyp_wordemb, src_wordemb, sub_embed, mult_embed), dim=2
        )
        # make sure to cut off gradients from padding, TODO rerun a model with this change
        # TODO use to be mt_attention_mask
        scores = self.estimator(embedded_sequences)*(mt_input_ids>2).unsqueeze(-1)
        
        # get number of tokens per line, de-factor bos, eos tokens
        avgnorm = torch.sum(mt_input_ids>2, 1).unsqueeze(-1)
        #print(avgnorm)
        # print(scores)

        # return token scores as well for inference
        # do average normalization for pooling
        return {"score": torch.sum(scores, 1)/avgnorm}

    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)
        df = df[["src", "mt", "score"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["score"] = df["score"].astype(float)
        return df.to_dict("records")

# Modified version, test for import code
class ReferencelessRegressionMod(RegressionMetric):
    """ReferencelessRegression:

    :param nr_frozen_epochs: Number of epochs (% of epoch) that the encoder is frozen.
    :param keep_embeddings_frozen: Keeps the encoder frozen during training.
    :param optimizer: Optimizer used during training.
    :param encoder_learning_rate: Learning rate used to fine-tune the encoder model.
    :param learning_rate: Learning rate used to fine-tune the top layers.
    :param layerwise_decay: Learning rate % decay from top-to-bottom encoder layers.
    :param encoder_model: Encoder model to be used.
    :param pretrained_model: Pretrained model from Hugging Face.
    :param pool: Pooling strategy to derive a sentence embedding ['cls', 'max', 'avg'].
    :param layer: Encoder layer to be used ('mix' for pooling info from all layers.)
    :param dropout: Dropout used in the top-layers.
    :param batch_size: Batch size used during training.
    :param train_data: Path to a csv file containing the training data.
    :param validation_data: Path to a csv file containing the validation data.
    :param hidden_sizes: Hidden sizes for the Feed Forward regression.
    :param activations: Feed Forward activation function.
    :param load_weights_from_checkpoint: Path to a checkpoint file.
    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = False,
        optimizer: str = "AdamW",
        encoder_learning_rate: float = 1e-05,
        learning_rate: float = 3e-05,
        layerwise_decay: float = 0.95,
        encoder_model: str = "XLM-RoBERTa",
        pretrained_model: str = "xlm-roberta-base",
        pool: str = "avg",
        layer: Union[str, int] = "mix",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: Optional[str] = None,
        validation_data: Optional[str] = None,
        hidden_sizes: List[int] = [1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        load_weights_from_checkpoint: Optional[str] = None,
    ) -> None:
        super(RegressionMetric, self).__init__(
            nr_frozen_epochs,
            keep_embeddings_frozen,
            optimizer,
            encoder_learning_rate,
            learning_rate,
            layerwise_decay,
            encoder_model,
            pretrained_model,
            pool,
            layer,
            dropout,
            batch_size,
            train_data,
            validation_data,
            load_weights_from_checkpoint,
            "referenceless_regression_metric",
        )
        self.save_hyperparameters()

    
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units * 4,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )
        """
        self.regressor = nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            # TODO expand this if we need to cat
            nn.Linear(self.encoder.model.config.hidden_size, 1), 
        )
        """
        
    
    def is_referenceless(self) -> bool:
        return True

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        src_inputs = self.encoder.prepare_sample(sample["src"])
        mt_inputs = self.encoder.prepare_sample(sample["mt"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        inputs = {**src_inputs, **mt_inputs}

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def forward(
        self,
        src_input_ids: torch.tensor,
        src_attention_mask: torch.tensor,
        mt_input_ids: torch.tensor,
        mt_attention_mask: torch.tensor,
        #mask_causal: False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        # form new input, make it faster with nograd?
        src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        # for inference time, if we need to pass in a custom mask
        #if mask_causal:
        #    hyp_wordemb = self.encoder(mt_input_ids, mt_attention_mask)['wordemb']
        #else:
        hyp_wordemb = self.encoder(mt_input_ids, causalmask(mt_attention_mask, dev=self.device))['wordemb']

        src_wordemb = src_sentemb[:, None, :].expand(hyp_wordemb.shape)
        sub_embed = hyp_wordemb - src_wordemb
        mult_embed = hyp_wordemb * src_wordemb

        # repr(x_1,...,x_n, y_i) = [y_i, pool(x_1,...,x_n), y_i - pool(x_1,...,x_n), y_i * pool(x_1,...,x_n)]
        # need to do that over i different output toks, is there an easy way to do that?
        embedded_sequences = torch.cat(
            (hyp_wordemb, src_wordemb, sub_embed, mult_embed), dim=2
        )
        
        scores = self.estimator(embedded_sequences)
        avgnorm = torch.sum(mt_attention_mask, 1).unsqueeze(-1)

        # return token scores as well for inference
        # do average normalization for pooling
        return {"score": torch.sum(scores, 1)/avgnorm, "tokscores":scores/avgnorm}

    def read_csv(self, path: str) -> List[dict]:
        """Reads a comma separated value file.

        :param path: path to a csv file.

        :return: List of records as dictionaries
        """
        df = pd.read_csv(path)
        df = df[["src", "mt", "score"]]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["score"] = df["score"].astype(float)
        return df.to_dict("records")
