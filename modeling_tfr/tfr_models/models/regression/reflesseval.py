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

# checkpoints to remember
# checkpoint 38 is the causal-less ablation
# checkpoint 34 is the causal COMET direct model
# use the comet-qe-20 checkpoint for training initialization

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
    
class ReflessEval(RegressionMetric):
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
        encoder_model: str = "RoBERTa", # TODO switch back in time of need
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
        mt_pos_ids: torch.tensor,
        mt_attention_mask: torch.tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        # make sure that the output mask is a custom causal mask given as input
        # TODO this code is probably wrong
        # encode string passed in for input
        if self.encoder_model=="BART":
            # TODO need to update later for our reflesseval, posid stuff
            src_sentemb, modelout = self.get_sentence_embedding(src_input_ids, src_attention_mask, mt_input_ids, mt_attention_mask, mt_pos_ids)
            hyp_wordemb = modelout['hypwordemb']
        else:
            # TODO do we need to disable causal mask for sanity?
            src_sentemb = self.get_sentence_embedding(src_input_ids, src_attention_mask)
        
            # encode hypothesis passed in, will use special pos_ids and attention mask passed in 
            hyp_wordemb = self.encoder(mt_input_ids, mt_attention_mask, position_ids=mt_pos_ids)['wordemb']

        # prepare format to go into feed-forward network (hyp tokens all kept separate)
        src_wordemb = src_sentemb[:, None, :].expand(hyp_wordemb.shape)
        sub_embed = hyp_wordemb - src_wordemb
        mult_embed = hyp_wordemb * src_wordemb

        # repr(x_1,...,x_n, y_i) = [y_i, pool(x_1,...,x_n), y_i - pool(x_1,...,x_n), y_i * pool(x_1,...,x_n)]
        embedded_sequences = torch.cat(
            (hyp_wordemb, src_wordemb, sub_embed, mult_embed), dim=2
        )
        # mask out scores related to padding, BOS token, # TODO validate if padding is done in the same way with bart
        scores = self.estimator(embedded_sequences)*(mt_input_ids>2).unsqueeze(-1)

        # compute something to normalize scores
        if mt_pos_ids is None:
            mask = mt_input_ids.ne(-1).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + 0) * mask
            mt_pos_ids = incremental_indices.long() - 1
            # add batch compatibility for normalization 
            mt_pos_ids = mt_pos_ids.expand(mt_input_ids.shape) 
        else:
            # we've been passed in lattice posids, or manual, still need to adjust
            mt_pos_ids = mt_pos_ids - 2
        mt_pos_ids = mt_pos_ids*(mt_input_ids>2)
        
        avgnorm = torch.max(mt_pos_ids, 1).values.unsqueeze(-1).unsqueeze(-1)
    
        # normalize scores with respect to highest position id (remove bos, eos, and extra 2 for xlm)
        scores = scores/avgnorm
        #print(torch.max(mt_pos_ids)-4)
        # TODO for efficient lattice, can undo norm (should be able to use exactly, in the same way)
        # return scores for each token as is, no need to normalize
        return {"score": scores, "norm":avgnorm}

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

