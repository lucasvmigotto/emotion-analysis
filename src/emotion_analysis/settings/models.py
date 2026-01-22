from pydantic import BaseModel


class ModelsSettings(BaseModel):
    MODEL_ID: str = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"

    FEATURE_EXTRACTOR_ID: str | None = None
