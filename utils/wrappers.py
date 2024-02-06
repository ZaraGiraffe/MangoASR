from dataclasses import dataclass
import torch
from transformers import WhisperForConditionalGeneration


@dataclass
class WhisperAsrWrapperConfig:
    version: int = 1
    pad_token_id = 50257


class WhisperAsrWrapperModel(torch.nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration,
                 config: WhisperAsrWrapperConfig = WhisperAsrWrapperConfig()):
        super().__init__()
        self.config = config
        self.model = model

    def forward(self, input_features, attention_mask, labels):
        model_output = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "loss": model_output.loss,
            "logits": model_output.logits,
            "labels": labels,
        }