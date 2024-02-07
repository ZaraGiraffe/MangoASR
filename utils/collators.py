from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from dataclasses import dataclass


@dataclass 
class WhisperEvalGenerator:
    processor: WhisperProcessor
    model: WhisperForConditionalGeneration
    language: str = "uk"
    task: str = "transcribe"

    def __call__(self, example: dict) -> str:
        """
        :param example: dictionary that has "audio":"array" key with sampling rate 16000
        :return asnwer: the string of predicted transcription
        """
        features_dict = self.processor.feature_extractor(
            example["audio"]["array"],
            return_tensors="pt",
            sampling_rate=16000
        )
        model_output = self.model.generate(**features_dict, language=self.language, task=self.task)
        tokens = model_output.tolist()[0]
        answer = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        return answer


@dataclass
class WhisperTrainCollator:
    """
    Collator function class, see __call__ method to analise the structure of the dataset
    that can be used with this collator
    :param processor: usual whisper processor
    :param device: this parameter determines where to put the output tensors
    """
    processor: WhisperProcessor
    device: str = "cpu"

    def __call__(self, raw_data: list[dict]) -> dict:
        """
        :param raw_data: list of dictionaries that have "audio":"array" key with sampling rate 16000
            and "transcription" key of the audio
        :return model_input: dict input to WhisperForConditionalGeneration for training
        """
        features_dict = self.processor.feature_extractor(
            [exm["audio"]["array"] for exm in raw_data], 
            return_tensors="pt",
            sampling_rate=16000
        )
        tokens_dict = self.processor.tokenizer([exm["transcription"] for exm in raw_data])
        max_length = max(list(map(lambda x: len(x), tokens_dict["input_ids"])))
        tokens_dict = {
            "input_ids": [v + [-100] * (max_length - len(v)) for v in tokens_dict["input_ids"]],
            "attention_mask": [v + [0] * (max_length - len(v)) for v in tokens_dict["attention_mask"]],
        }
        model_input = {
            **features_dict,
            "labels": torch.tensor(tokens_dict["input_ids"]),
            "attention_mask": torch.tensor(tokens_dict["attention_mask"]),
        }
        model_input = {k: v.to(self.device) for k, v in model_input.items()}
        return model_input