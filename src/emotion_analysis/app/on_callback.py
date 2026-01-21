from pathlib import Path

from ..services import Classifier


def on_classify(model: Classifier):
    def fn(audio: Path | str):
        probs: dict[int, float] = model.predict(audio, return_probs=True)  # type: ignore
        print(probs)
        return list(probs.values())

    return fn
