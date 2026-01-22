from logging import basicConfig, getLogger
from os import environ, getenv

from ..settings import LogSettings


def setup_log(settings: LogSettings | None = None):
    _settings: LogSettings = settings or LogSettings()
    basicConfig(**_settings.config)
    for mod, level in _settings.SUPPRESS_MODULES:
        getLogger(mod).setLevel(level or _settings.SUPPRESS_LEVEL)


def setup_envvars(*envs: dict[str, str]) -> None:
    for env in envs:
        for env_key, env_value in env.items():
            if not getenv(env_key):
                environ[env_key] = env_value
