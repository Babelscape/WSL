from typing import Any, Dict, Optional

import torch
from transformers import AutoModel, PreTrainedModel
from transformers.activations import ClippedGELUActivation, GELUActivation
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PoolerEndLogits

from .configuration_wsl import WSLReaderConfig


class WSLeaderSample:
    def __init__(self, **kwargs):
        super().__setattr__("_d", {})
        self._d = kwargs

    def __getattribute__(self, item):
        return super(WSLeaderSample, self).__getattribute__(item)

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            # this is likely some python library-specific variable (such as __deepcopy__ for copy)
            # better follow standard behavior here
            raise AttributeError(item)
        elif item in self._d:
            return self._d[item]
        else:
            return None

    def __setattr__(self, key, value):
        if key in self._d:
            self._d[key] = value
        else:
            super().__setattr__(key, value)
            self._d[key] = value


activation2functions = {
    "relu": torch.nn.ReLU(),
    "gelu": GELUActivation(),
    "gelu_10": ClippedGELUActivation(-10, 10),
}


class PoolerEndLogitsBi(PoolerEndLogits):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dense_1 = torch.nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if p_mask is not None:
            p_mask = p_mask.unsqueeze(-1)
        logits = super().forward(
            hidden_states,
            start_states,
            start_positions,
            p_mask,
        )
        return logits


class WSLReaderSpanModel(PreTrainedModel):
    config_class = WSLReaderConfig

    def __init__(self, config: WSLReaderConfig, *args, **kwargs):
        super().__init__(config)
        # Transformer model declaration
        self.config = config
        self.transformer_model = (
            AutoModel.from_pretrained(self.config.transformer_model)
            if self.config.num_layers is None
            else AutoModel.from_pretrained(
                self.config.transformer_model, num_hidden_layers=self.config.num_layers
            )
        )
        self.transformer_model.resize_token_embeddings(
            self.transformer_model.config.vocab_size
            + self.config.additional_special_symbols
        )

        self.activation = self.config.activation
        self.linears_hidden_size = self.config.linears_hidden_size
        self.use_last_k_layers = self.config.use_last_k_layers

        # named entity detection layers
        self.ned_start_classifier = self._get_projection_layer(
            self.activation, last_hidden=2, layer_norm=False
        )
        if self.config.binary_end_logits:
            self.ned_end_classifier = PoolerEndLogitsBi(self.transformer_model.config)
        else:
            self.ned_end_classifier = PoolerEndLogits(self.transformer_model.config)

        # END entity disambiguation layer
        self.ed_start_projector = self._get_projection_layer(self.activation)
        self.ed_end_projector = self._get_projection_layer(self.activation)

        self.training = self.config.training

        # criterion
        self.criterion = torch.nn.CrossEntropyLoss()

    def _get_projection_layer(
        self,
        activation: str,
        last_hidden: Optional[int] = None,
        input_hidden=None,
        layer_norm: bool = True,
    ) -> torch.nn.Sequential:
        head_components = [
            torch.nn.Dropout(0.1),
            torch.nn.Linear(
                (
                    self.transformer_model.config.hidden_size * self.use_last_k_layers
                    if input_hidden is None
                    else input_hidden
                ),
                self.linears_hidden_size,
            ),
            activation2functions[activation],
            torch.nn.Dropout(0.1),
            torch.nn.Linear(
                self.linears_hidden_size,
                self.linears_hidden_size if last_hidden is None else last_hidden,
            ),
        ]

        if layer_norm:
            head_components.append(
                torch.nn.LayerNorm(
                    self.linears_hidden_size if last_hidden is None else last_hidden,
                    self.transformer_model.config.layer_norm_eps,
                )
            )

        return torch.nn.Sequential(*head_components)

    def _mask_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        if next(self.parameters()).dtype == torch.float16:
            logits = logits * (1 - mask) - 65500 * mask
        else:
            logits = logits * (1 - mask) - 1e30 * mask
        return logits

    def _get_model_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
    ):
        model_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": self.use_last_k_layers > 1,
        }

        if token_type_ids is not None:
            model_input["token_type_ids"] = token_type_ids

        model_output = self.transformer_model(**model_input)

        if self.use_last_k_layers > 1:
            model_features = torch.cat(
                model_output[1][-self.use_last_k_layers :], dim=-1
            )
        else:
            model_features = model_output[0]

        return model_features

    def compute_ned_end_logits(
        self,
        start_predictions,
        start_labels,
        model_features,
        prediction_mask,
        batch_size,
    ) -> Optional[torch.Tensor]:
        # todo: maybe when constraining on the spans,
        #  we should not use a prediction_mask for the end tokens.
        #  at least we should not during training imo
        start_positions = start_labels if self.training else start_predictions
        start_positions_indices = (
            torch.arange(start_positions.size(1), device=start_positions.device)
            .unsqueeze(0)
            .expand(batch_size, -1)[start_positions > 0]
        ).to(start_positions.device)

        if len(start_positions_indices) > 0:
            expanded_features = model_features.repeat_interleave(
                torch.sum(start_positions > 0, dim=-1), dim=0
            )
            expanded_prediction_mask = prediction_mask.repeat_interleave(
                torch.sum(start_positions > 0, dim=-1), dim=0
            )
            end_logits = self.ned_end_classifier(
                hidden_states=expanded_features,
                start_positions=start_positions_indices,
                p_mask=expanded_prediction_mask,
            )

            return end_logits

        return None

    def compute_classification_logits(
        self,
        model_features_start,
        model_features_end,
        special_symbols_features,
    ) -> torch.Tensor:
        model_start_features = self.ed_start_projector(model_features_start)
        model_end_features = self.ed_end_projector(model_features_end)
        model_start_features_symbols = self.ed_start_projector(special_symbols_features)
        model_end_features_symbols = self.ed_end_projector(special_symbols_features)

        model_ed_features = torch.cat(
            [model_start_features, model_end_features], dim=-1
        )
        special_symbols_representation = torch.cat(
            [model_start_features_symbols, model_end_features_symbols], dim=-1
        )

        logits = torch.bmm(
            model_ed_features,
            torch.permute(special_symbols_representation, (0, 2, 1)),
        )

        logits = self._mask_logits(logits, (model_features_start == -100).all(2).long())
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        prediction_mask: Optional[torch.Tensor] = None,
        special_symbols_mask: Optional[torch.Tensor] = None,
        start_labels: Optional[torch.Tensor] = None,
        end_labels: Optional[torch.Tensor] = None,
        use_predefined_spans: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape

        model_features = self._get_model_features(
            input_ids, attention_mask, token_type_ids
        )

        ned_start_labels = None

        # named entity detection if required
        if use_predefined_spans:  # no need to compute spans
            ned_start_logits, ned_start_probabilities, ned_start_predictions = (
                None,
                None,
                (
                    torch.clone(start_labels)
                    if start_labels is not None
                    else torch.zeros_like(input_ids)
                ),
            )
            ned_end_logits, ned_end_probabilities, ned_end_predictions = (
                None,
                None,
                (
                    torch.clone(end_labels)
                    if end_labels is not None
                    else torch.zeros_like(input_ids)
                ),
            )
            ned_start_predictions[ned_start_predictions > 0] = 1
            ned_end_predictions[end_labels > 0] = 1
            ned_end_predictions = ned_end_predictions[~(end_labels == -100).all(2)]

        else:  # compute spans
            # start boundary prediction
            ned_start_logits = self.ned_start_classifier(model_features)
            ned_start_logits = self._mask_logits(ned_start_logits, prediction_mask)
            ned_start_probabilities = torch.softmax(ned_start_logits, dim=-1)
            ned_start_predictions = ned_start_probabilities.argmax(dim=-1)

            # end boundary prediction
            ned_start_labels = (
                torch.zeros_like(start_labels) if start_labels is not None else None
            )

            if ned_start_labels is not None:
                ned_start_labels[start_labels == -100] = -100
                ned_start_labels[start_labels > 0] = 1

            ned_end_logits = self.compute_ned_end_logits(
                ned_start_predictions,
                ned_start_labels,
                model_features,
                prediction_mask,
                batch_size,
            )

            if ned_end_logits is not None:
                ned_end_probabilities = torch.softmax(ned_end_logits, dim=-1)
                if not self.config.binary_end_logits:
                    ned_end_predictions = torch.argmax(
                        ned_end_probabilities, dim=-1, keepdim=True
                    )
                    ned_end_predictions = torch.zeros_like(
                        ned_end_probabilities
                    ).scatter_(1, ned_end_predictions, 1)
                else:
                    ned_end_predictions = torch.argmax(ned_end_probabilities, dim=-1)
            else:
                ned_end_logits, ned_end_probabilities = None, None
                ned_end_predictions = ned_start_predictions.new_zeros(
                    batch_size, seq_len
                )

            if not self.training:
                # if len(ned_end_predictions.shape) < 2:
                #     print(ned_end_predictions)
                end_preds_count = ned_end_predictions.sum(1)
                # If there are no end predictions for a start prediction, remove the start prediction
                if (end_preds_count == 0).any() and (ned_start_predictions > 0).any():
                    ned_start_predictions[ned_start_predictions == 1] = (
                        end_preds_count != 0
                    ).long()
                    ned_end_predictions = ned_end_predictions[end_preds_count != 0]

        if end_labels is not None:
            end_labels = end_labels[~(end_labels == -100).all(2)]

        start_position, end_position = (
            (start_labels, end_labels)
            if self.training
            else (ned_start_predictions, ned_end_predictions)
        )
        start_counts = (start_position > 0).sum(1)
        if (start_counts > 0).any():
            ned_end_predictions = ned_end_predictions.split(start_counts.tolist())
        # Entity disambiguation
        if (end_position > 0).sum() > 0:
            ends_count = (end_position > 0).sum(1)
            model_entity_start = torch.repeat_interleave(
                model_features[start_position > 0], ends_count, dim=0
            )
            model_entity_end = torch.repeat_interleave(
                model_features, start_counts, dim=0
            )[end_position > 0]
            ents_count = torch.nn.utils.rnn.pad_sequence(
                torch.split(ends_count, start_counts.tolist()),
                batch_first=True,
                padding_value=0,
            ).sum(1)

            model_entity_start = torch.nn.utils.rnn.pad_sequence(
                torch.split(model_entity_start, ents_count.tolist()),
                batch_first=True,
                padding_value=-100,
            )

            model_entity_end = torch.nn.utils.rnn.pad_sequence(
                torch.split(model_entity_end, ents_count.tolist()),
                batch_first=True,
                padding_value=-100,
            )

            ed_logits = self.compute_classification_logits(
                model_entity_start,
                model_entity_end,
                model_features[special_symbols_mask].view(
                    batch_size, -1, model_features.shape[-1]
                ),
            )
            ed_probabilities = torch.softmax(ed_logits, dim=-1)
            ed_predictions = torch.argmax(ed_probabilities, dim=-1)
        else:
            ed_logits, ed_probabilities, ed_predictions = (
                None,
                ned_start_predictions.new_zeros(batch_size, seq_len),
                ned_start_predictions.new_zeros(batch_size),
            )
        # output build
        output_dict = dict(
            batch_size=batch_size,
            ned_start_logits=ned_start_logits,
            ned_start_probabilities=ned_start_probabilities,
            ned_start_predictions=ned_start_predictions,
            ned_end_logits=ned_end_logits,
            ned_end_probabilities=ned_end_probabilities,
            ned_end_predictions=ned_end_predictions,
            ed_logits=ed_logits,
            ed_probabilities=ed_probabilities,
            ed_predictions=ed_predictions,
        )

        # compute loss if labels
        if start_labels is not None and end_labels is not None and self.training:
            # named entity detection loss

            # start
            if ned_start_logits is not None:
                ned_start_loss = self.criterion(
                    ned_start_logits.view(-1, ned_start_logits.shape[-1]),
                    ned_start_labels.view(-1),
                )
            else:
                ned_start_loss = 0

            # end
            # use ents_count to assign the labels to the correct positions i.e. using end_labels -> [[0,0,4,0], [0,0,0,2]] -> [4,2] (this is just an element, for batch we need to mask it with ents_count), ie -> [[4,2,-100,-100], [3,1,2,-100], [1,3,2,5]]

            if ned_end_logits is not None:
                ed_labels = end_labels.clone()
                ed_labels = torch.nn.utils.rnn.pad_sequence(
                    torch.split(ed_labels[ed_labels > 0], ents_count.tolist()),
                    batch_first=True,
                    padding_value=-100,
                )
                end_labels[end_labels > 0] = 1
                if not self.config.binary_end_logits:
                    # transform label to position in the sequence
                    end_labels = end_labels.argmax(dim=-1)
                    ned_end_loss = self.criterion(
                        ned_end_logits.view(-1, ned_end_logits.shape[-1]),
                        end_labels.view(-1),
                    )
                else:
                    ned_end_loss = self.criterion(
                        ned_end_logits.reshape(-1, ned_end_logits.shape[-1]),
                        end_labels.reshape(-1).long(),
                    )

                # entity disambiguation loss
                ed_loss = self.criterion(
                    ed_logits.view(-1, ed_logits.shape[-1]),
                    ed_labels.view(-1).long(),
                )

            else:
                ned_end_loss = 0
                ed_loss = 0

            output_dict["ned_start_loss"] = ned_start_loss
            output_dict["ned_end_loss"] = ned_end_loss
            output_dict["ed_loss"] = ed_loss

            output_dict["loss"] = ned_start_loss + ned_end_loss + ed_loss

        return output_dict
