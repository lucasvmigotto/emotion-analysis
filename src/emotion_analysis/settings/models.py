from pydantic import BaseModel


class ModelsSettings(BaseModel):
    MODEL_ID: str = "lucasvmigotto/whisper-small-audio-emotion-classification"

    FEATURE_EXTRACTOR_ID: str = "openai/whisper-small"
