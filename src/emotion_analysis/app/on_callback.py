from pathlib import Path

from gradio import Error

from ..services import Classifier


def on_classify(model: Classifier):
    def fn(audio: Path | str):
        if not audio:
            Error("No Audio selected")
            return [None] * len(model.id2label.keys())
        return [f"{prob:.03%}" for prob in model.predict(audio)]

    return fn
