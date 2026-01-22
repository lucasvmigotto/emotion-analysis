from typing import Self

from pydantic import BaseModel, computed_field


class LogSettings(BaseModel):
    LEVEL: str = "INFO"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    FORMAT: str = "{asctime} {levelname} {name}.{funcName}: {message}"
    STYLE: str = "{"

    SUPPRESS_MODULES: set[tuple[str, str | None]] = {
        ("urllib3", None),
        ("asyncio", None),
        ("httpcore", None),
        ("httpx", None),
        ("numba", None),
        ("python_multipart", None),
    }
    SUPPRESS_LEVEL: str = "ERROR"

    @computed_field
    @property
    def config(self: Self) -> dict[str, str]:
        return {
            "level": self.LEVEL,
            "datefmt": self.DATE_FORMAT,
            "format": self.FORMAT,
            "style": self.STYLE,
        }
