from pydantic_settings import BaseSettings, SettingsConfigDict

from .gradio import GradioSettings
from .hf import HuggingFaceSettings
from .log import LogSettings
from .models import ModelsSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_ignore_empty=True,
        extra="ignore",
        env_prefix="EMOTION_ANALYSIS__",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    GRADIO: GradioSettings = GradioSettings()
    MODELS: ModelsSettings = ModelsSettings()
    LOG: LogSettings = LogSettings()
    HF: HuggingFaceSettings = HuggingFaceSettings()
