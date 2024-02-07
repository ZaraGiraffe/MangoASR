from dataclasses import dataclass
import torch
from transformers import WhisperForConditionalGeneration


@dataclass
class WhisperAsrWrapperConfig:
    version: int = 1
    pad_token_id: int = -100


class WhisperAsrWrapperModel(torch.nn.Module):
    """
    The class that wraps WhisperForConditionalGeneration model for training purposes
    This class only wraps slightly forward method in order to make the output in
    the given format
    """
    def __init__(self, model: WhisperForConditionalGeneration,
                 config: WhisperAsrWrapperConfig = WhisperAsrWrapperConfig()):
        """
        :param model: the model, usually with pretrained weights
        :param config: (NOT OPTIONAL!) some data from here is needed for trainer, other just for configs
        """
        super().__init__()
        self.config = config
        self.model = model

    def forward(self, input_features, attention_mask, labels) -> dict:
        """
        :params input_features, attention_mask, labels: the parameters, that are required for
            WhisperForConditionalGeneration.forward (training regime)
        :return dictionary: the output has "loss" for updating weights,
            "predictions" and "labels" for calculating metrics
        """
        model_output = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": model_output.loss,
            "predictions": torch.argmax(model_output.logits, dim=2),
            "labels": labels,
        }