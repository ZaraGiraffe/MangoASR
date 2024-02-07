import torch
from transformers import WhisperProcessor
from typing import Callable
from .trainers import TrainingOutput
from dataclasses import dataclass


def compute_wer_loss(wer: Callable[[list[str], list[str]], float], predictions: torch.Tensor, labels: torch.Tensor, processor: WhisperProcessor) -> float:
    """
    :param wer: it is the instance from evaluate.load("wer") call
    :param predictions, labels: predictions and labels tensors of tokens respectively
    :param processor: the WhisperProcessor that is used for decoding the tokens
    """
    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    labels_str = processor.batch_decode(labels, skip_special_tokens=True)
    error = wer.compute(predictions=pred_str, references=labels_str)
    return error


def compute_cer_loss(cer: Callable[[list[str], list[str]], float], predictions: torch.Tensor, labels: torch.Tensor, processor: WhisperProcessor) -> float:
    """
    :param cer: it is the instance from evaluate.load("cer") call
    :param predictions, labels: predictions and labels tensors of tokens respectively
    :param processor: the WhisperProcessor that is used for decoding the tokens
    """
    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    labels_str = processor.batch_decode(labels, skip_special_tokens=True)
    error = cer.compute(predictions=pred_str, references=labels_str)
    return error


@dataclass
class ComputeStringSimilarityMetricsFunction:
    """
    This class is just a wrapper for a training cycle
    It wraps compute_wer_loss and compute_cer_loss to compute wer and cer losses
    :params wer, cer: are instances of evaluate.load("wer") and evaluate.load("cer")
    """
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