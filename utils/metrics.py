import torch
from transformers import WhisperProcessor
from typing import Callable
from .trainers import TrainingOutput
from dataclasses import dataclass




def compute_wer_loss(wer: Callable[[list[str], list[str]], float], predictions: torch.Tensor,
                     labels: torch.Tensor, processor: WhisperProcessor, pad_token_id: int) -> float:
    """
    :param wer: it is the instance from evaluate.load("wer") call
    :param predictions, labels: predictions and labels tensors of tokens respectively
    :param processor: the WhisperProcessor that is used for decoding the tokens
    :param pad_token_id: convert all negative values in predictions and labels to this value
    """
    mask = predictions < 0
    predictions[mask] = pad_token_id
    mask = labels < 0
    labels[mask] = pad_token_id

    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    labels_str = processor.batch_decode(labels, skip_special_tokens=True)
    error = wer.compute(predictions=pred_str, references=labels_str)
    return error


def compute_cer_loss(cer: Callable[[list[str], list[str]], float], predictions: torch.Tensor,
                     labels: torch.Tensor, processor: WhisperProcessor, pad_token_id: int) -> float:
    """
    :param cer: it is the instance from evaluate.load("cer") call
    :param predictions, labels: predictions and labels tensors of tokens respectively
    :param processor: the WhisperProcessor that is used for decoding the tokens
    """
    mask = predictions < 0
    predictions[mask] = pad_token_id
    mask = labels < 0
    labels[mask] = pad_token_id

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
    :param pad_token_id: if there are negative tokens (pad tokens) in the tensors, it will convert
        them to pad_token_id
    """
    processor: WhisperProcessor
    cer: Callable[[list[str], list[str]], float]
    wer: Callable[[list[str], list[str]], float]
    pad_token_id: int = 50257

    def __call__(self, output: TrainingOutput):
        wer = compute_wer_loss(self.wer, output.model_outputs["predictions"],
                               output.model_outputs["labels"], self.processor, pad_token_id=self.pad_token_id)
        cer = compute_wer_loss(self.cer, output.model_outputs["predictions"],
                               output.model_outputs["labels"], self.processor, pad_token_id=self.pad_token_id)
        return {
            "wer": wer,
            "cer": cer,
        }