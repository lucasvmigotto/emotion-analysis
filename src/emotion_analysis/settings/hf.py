from typing import Self

from pydantic import BaseModel, computed_field


class HuggingFaceSettings(BaseModel):
    HOME: str = "/hf"

    TOKEN: str | None = None

    @computed_field
    @property
    def env(self: Self) -> dict[str, str]:
        return {
            "HF_HOME": self.HOME,
            "HF_TOKEN": self.TOKEN
        }
