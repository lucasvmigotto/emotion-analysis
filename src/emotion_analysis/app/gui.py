from typing import Callable

from gradio import Audio, Blocks, Button, Column, Label, Row

from ..services import Classifier


def build_gui(model: Classifier, on_btn_classify: Callable):
    with Blocks() as demo:
        with Column():
            audio_input = Audio(
                label="Audio input",
                sources=["microphone", "upload"],
                type="filepath",
            )
            classify_btn = Button(value="Classify")

        with Row():
            emotions = [
                Label(label=emotion.capitalize()) for emotion in model.label2id.keys()
            ]

        classify_btn.click(
            fn=on_btn_classify(model), inputs=audio_input, outputs=emotions
        )

        return demo
