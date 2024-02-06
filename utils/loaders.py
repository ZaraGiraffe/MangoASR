import datasets
from datasets import load_dataset


def get_common_voice(lang, streaming=False) -> datasets.DatasetDict:
    """
    Loads Ukrainian Common Voice dataset from here
    https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0
    makes it compatible with DatasetMixer
    :param lang: the language of common voice dataset. Should be in ["en", "uk"]
    :param streaming (optional): pass this parameter to load_dataset function 
    """
    uk_speech_dataset = load_dataset("mozilla-foundation/common_voice_11_0", lang, streaming=streaming)
    uk_speech_dataset = uk_speech_dataset.rename_columns({"client_id": "speaker_id", "sentence": "transcription"})
    return uk_speech_dataset


def get_urban_sound() -> datasets.Dataset:
    """
    Loads Urbanban Sounde noise dataset from here
    https://huggingface.co/datasets/danavery/urbansound8K
    makes it compatible with DatasetMixer
    """
    noise_dataset = load_dataset("danavery/urbansound8K")
    column = noise_dataset["train"]["class"]
    noise_dataset = noise_dataset["train"].add_column(name="label", column=column)
    return noise_dataset
