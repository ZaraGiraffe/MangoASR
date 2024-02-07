import torch
from transformers import WhisperProcessor
from typing import Callable
from .trainers import TrainingOutput
from dataclasses import dataclass


def compute_wer_loss(wer: Callable[[list[str], list[str]], float], predictions: torch.Tensor, labels: torch.Tensor, processor: WhisperProcessor) -> float:
    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    labels_str = processor.batch_decode(labels, skip_special_tokens=True)
    error = wer.compute(predictions=pred_str, references=labels_str)
    return error


def compute_cer_loss(cer: Callable[[list[str], list[str]], float], predictions: torch.Tensor, labels: torch.Tensor, processor: WhisperProcessor) -> float:
    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    labels_str = processor.batch_decode(labels, skip_special_tokens=True)
    error = cer.compute(predictions=pred_str, references=labels_str)
    return error


@dataclass
class ComputeStringSimilarityMetricsFunction:
    processor: WhisperProcessor
    cer: Callable[[list[str], list[str]], float]
    wer: Callable[[list[str], list[str]], float]

    def __call__(self, output: TrainingOutput):
        wer = compute_wer_loss(self.wer, output.model_outputs["predictions"], output.model_outputs["labels"], self.processor)
        cer = compute_wer_loss(self.cer, output.model_outputs["predictions"], output.model_outputs["labels"], self.processor)
        return {
            "wer": wer,
            "cer": cer,
        }