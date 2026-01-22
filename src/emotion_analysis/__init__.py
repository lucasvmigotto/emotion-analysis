from logging import Logger, getLogger

from .app import build_gui, on_classify
from .services import Classifier
from .settings import Settings
from .utils import setup_envvars, setup_log


def main(settings: Settings, /, logger: Logger | None = None):
    _logger = logger or getLogger(__name__)

    setup_envvars(settings.GRADIO.env)
    setup_log(settings.LOG)

    _logger.debug("Initiating model")
    model = Classifier(settings.MODELS)
    _logger.info("Model ready")

    _logger.debug("Building and launching app")
    build_gui(model, on_btn_classify=on_classify).launch(**settings.GRADIO.config)


if __name__ == "__main__":
    main(Settings())
