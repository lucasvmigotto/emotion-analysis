from pathlib import Path
from typing import Literal

import gradio as gr
from librosa import load as librosa_load
from numpy import pad as np_pad
from torch import Tensor as TorchTensor
from torch import device as torch_device
from torch import inference_mode as torch_inference_mode
from torch.cuda import is_available as torch_cuda_is_available
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from typing_extensions import Self

_TensorsReturnType = Literal["pt", "tf"]
MODEL_ID: str = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"


EMOJI_EMOTIONS: dict[str, str] = {
    "angry": "😡",
    "disgust": "🤮",
    "fearful": "😱",
    "happy": "😄",
    "neutral": "😶",
    "sad": "😥",
    "surprised": "😮",
}

i18n = gr.I18n(
    en={
        "sad": "Triste",
        "happy": "Feliz",
        "angry": "Irritado",
        "neutral": "Neutro",
        "disgust": "Desgosto",
        "fearful": "Amedrontado",
        "surprised": "Surpreso",
        "empty_emotion": "🫥 Emoção",
        "audio_input": "Áudio para ser classificado",
        "classify_btn": "Classificar",
    }
)


class AudioEmotion:
    def _load_model(self: Self) -> Self:
        self._model = AutoModelForAudioClassification.from_pretrained(self._model_id)
        self._id_to_label: dict[int, str] = self._model.config.id2label
        self._model.to(self._device)
        return self

    def _load_feature_extractor(self: Self) -> Self:
        self._feat_extractor = AutoFeatureExtractor.from_pretrained(self._model_id)
        return self

    def __init__(
        self: Self,
        model_id: str,
        /,
        audio_max_duration: float | int | None = None,
        tensors_return_type: _TensorsReturnType | None = None,
        truncate: bool = True,
    ):
        self._device = torch_device("cuda" if torch_cuda_is_available() else "cpu")
        self._model_id: str = model_id

        self._model: AutoModelForAudioClassification | None = None
        self._id_to_label: dict[int, str] | None = None
        self._feat_extractor: AutoFeatureExtractor | None = None

        self._audio_max_duration = audio_max_duration or 30.0
        self._tensors_return_type = tensors_return_type or "pt"
        self._truncate = truncate

        self._load_model()._load_feature_extractor()

    def __repr__(self: Self) -> str:
        return (
            "Model (on: {device}): {model}\nFeature Extractor: {feat_extract}".format(
                device=self._device,
                model=self._model.__repr__(),
                feat_extract=self._feat_extractor.__repr__(),
            )
        )

    def _preprocess(self: Self, audio_path: Path | str) -> dict[str, TorchTensor]:
        audio, _ = librosa_load(audio_path, sr=None)

        if (audio_length := len(audio)) > (
            max_length := int(
                self._feat_extractor.sampling_rate * self._audio_max_duration
            )
        ):
            audio = audio[:max_length]
        else:
            audio = np_pad(audio, (0, max_length - audio_length))

        inputs: dict[str, TorchTensor] = self._feat_extractor(
            audio,
            sampling_rate=self._feat_extractor.sampling_rate,
            max_length=max_length,
            truncation=self._truncate,
            return_tensors=self._tensors_return_type,
        )

        return {key: value.to(self._device) for key, value in inputs.items()}

    def predict(
        self: Self, audio_path: Path | str, /, return_probs: bool = False
    ) -> dict[str, float] | str:
        inputs: dict[str, TorchTensor] = self._preprocess(audio_path)

        with torch_inference_mode():
            y_hat: TorchTensor = self._model(**inputs).logits.cpu()

        if return_probs:
            return {
                self._id_to_label.get(idx): prob.item()
                for idx, prob in enumerate(y_hat.softmax(dim=-1)[0])
            }

        return self._id_to_label.get(y_hat.argmax(dim=-1).item())


model = AudioEmotion(MODEL_ID)


def markdown_mask(text: str, width: int = 1) -> str:
    return f"<center><h{width}>{text}</h{width}></center>"


def classify(audio_input_path: Path | str) -> list[gr.Markdown]:
    probs: dict[str, float] | str = model.predict(audio_input_path, return_probs=True)

    return [
        gr.Markdown(
            markdown_mask(f"{EMOJI_EMOTIONS.get(label)} {i18n(label)} {prob:.02%}")
        )
        for label, prob in probs.items()
    ]


with gr.Blocks() as demo:
    audio_input = gr.Audio(
        label=i18n("audio_input"),
        sources=["microphone", "upload"],
        type="filepath",
        show_download_button=False,
    )
    classify_btn = gr.Button(i18n("classify_btn"))

    with gr.Column():
        emotions = [
            gr.Markdown(markdown_mask(f"{emoji} {i18n(label)} {0.0:.02%}"))
            for label, emoji in EMOJI_EMOTIONS.items()
        ]

    classify_btn.click(classify, inputs=audio_input, outputs=emotions)

if __name__ == "__main__":
    demo.launch(share=False, show_api=False, i18n=i18n)
