from .app import build_gui, on_classify
from .services import Classifier
from .settings import Settings
from .utils import setup_envvars, setup_log


def main(settings: Settings):
    setup_envvars(settings.GRADIO.env, settings.HF.env)
    setup_log(settings.LOG)

    model = Classifier(settings.MODELS)

    build_gui(model, on_btn_classify=on_classify).launch(settings.GRADIO.config)


if __name__ == "__main__":
    main(Settings())
