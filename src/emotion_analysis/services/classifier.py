from logging import Logger, getLogger
from pathlib import Path
from typing import Literal, Self

from librosa import load as librosa_load
from numpy import pad as np_pad
from torch import Tensor
from torch import device as torch_device
from torch import inference_mode as torch_inferece_mode
from torch.cuda import is_available as cuda_is_available
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from ..settings import ModelsSettings
from ..utils import timeit


class Classifier:
    def __init__(
        self: Self,
        settings: ModelsSettings,
        /,
        audio_max_duration: int | None = 30,
    ):
        self._log: Logger = getLogger(__name__)

        self._log.debug("Selecting model device")
        self._device = torch_device("cuda" if cuda_is_available() else "cpu")
        self._log.info(f"Model will be initiated in {self._device}")

        self._log.debug(f"Get pretrained {settings.MODEL_ID} to {self._device}")
        self._model = AutoModelForAudioClassification.from_pretrained(
            settings.MODEL_ID
        ).to(self._device)
        self._log.info(f"Model {settings.MODEL_ID} available in {self._device}")

        self._log.debug(
            f"Get feature extractor {settings.FEATURE_EXTRACTOR_ID or settings.MODEL_ID}"
        )
        self._feat_extractor = AutoFeatureExtractor.from_pretrained(
            settings.FEATURE_EXTRACTOR_ID or settings.MODEL_ID
        )
        self._max_length = int(self._feat_extractor.sampling_rate * audio_max_duration)

    @property
    def id2label(self: Self) -> dict[int, str]:
        return self._model.config.id2label

    @property
    def label2id(self: Self) -> dict[str, int]:
        return self._model.config.label2id

    @timeit
    def _preprocess(
        self: Self,
        audio_path: Path | str,
        /,
        truncation: bool = True,
        return_tensors_type: Literal["pt", "tf"] = "pt",
    ) -> dict[str, Tensor]:
        self._log.debug(f"Preprocessing audio: {audio_path}")

        audio, _ = librosa_load(audio_path, sr=None)
        audio = (
            audio[: self._max_length]
            if len(audio) > self._max_length
            else np_pad(audio, (0, self._max_length))
        )
        return self._feat_extractor(
            audio,
            sampling_rate=self._feat_extractor.sampling_rate,
            max_length=self._max_length,
            truncation=truncation,
            return_tensors=return_tensors_type,
        )

    @timeit
    def _predict(self: Self, audio_sample: dict[str, Tensor]) -> Tensor:
        with torch_inferece_mode():
            prediction: SequenceClassifierOutput = self._model(**audio_sample)
            self._log.debug(f"Prediction logits: {prediction.logits}")
        return prediction.logits

    def predict(
        self: Self,
        audio: Path | str,
        /,
        return_labeled_probs: bool = False,
    ) -> dict[int, float] | Tensor:
        probs: Tensor = (
            self._predict(
                {
                    key: value.to(self._device)
                    for key, value in self._preprocess(audio).items()
                }
            )
            .cpu()
            .softmax(dim=-1)[0]
        )

        self._log.info(f"Prediction probabilities: {probs}")

        return (
            {idx: prob.item() for idx, prob in enumerate(probs)}
            if return_labeled_probs
            else probs
        )
