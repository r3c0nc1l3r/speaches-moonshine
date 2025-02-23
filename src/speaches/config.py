from __future__ import annotations

import enum

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SAMPLES_PER_SECOND = 16000
BYTES_PER_SAMPLE = 2
BYTES_PER_SECOND = SAMPLES_PER_SECOND * BYTES_PER_SAMPLE
# 2 BYTES = 16 BITS = 1 SAMPLE
# 1 SECOND OF AUDIO = 32000 BYTES = 16000 SAMPLES


class ResponseFormat(enum.StrEnum):
    TEXT = "text"
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    SRT = "srt"
    VTT = "vtt"


class Language(str):
    """Language code in ISO 639-1 format."""


class Task(enum.StrEnum):
    """Task to perform."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class Device(enum.StrEnum):
    """Device to use for inference."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class Quantization(enum.StrEnum):
    """Quantization type to use for inference."""

    DEFAULT = "default"
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    INT16 = "int16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class MoonshineConfig(BaseModel):
    """Configuration for Moonshine ASR."""

    model_path: str = Field(default="UsefulSensors/moonshine")
    """
    Path to the Moonshine model. Can be either:
    1. A HuggingFace model ID (e.g. 'UsefulSensors/moonshine' or 'UsefulSensors/moonshine:main')
    2. A local directory containing the model files
    3. A direct path to the model file
    """

    ttl: int = Field(default=300, ge=-1)
    """
    Time in seconds until the model is unloaded if it is not being used.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """


class WhisperConfig(BaseModel):
    """See https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L599."""

    model: str = Field(default="Systran/faster-whisper-small")
    """
    Default HuggingFace model to use for transcription. Note, the model must support being ran using CTranslate2.
    This model will be used if no model is specified in the request.

    Models created by authors of `faster-whisper` can be found at https://huggingface.co/Systran
    You can find other supported models at https://huggingface.co/models?p=2&sort=trending&search=ctranslate2 and https://huggingface.co/models?sort=trending&search=ct2
    """
    inference_device: Device = Field(default=Device.AUTO)
    device_index: int | list[int] = 0
    compute_type: Quantization = Field(default=Quantization.DEFAULT)
    cpu_threads: int = 0
    num_workers: int = 1
    ttl: int = Field(default=300, ge=-1)
    """
    Time in seconds until the model is unloaded if it is not being used.
    -1: Never unload the model.
    0: Unload the model immediately after usage.
    """
    use_batched_mode: bool = False
    """
    Whether to use batch mode(introduced in 1.1.0 `faster-whisper` release) for inference. This will likely become the default in the future and the configuration option will be removed.
    """


class Config(BaseSettings):
    """Configuration for the application. Values can be set via environment variables."""

    model_config = SettingsConfigDict(env_prefix="SPEACHES_")

    api_key: str | None = None
    """API key for authentication. If not set, authentication is disabled."""

    default_language: Language = "en"
    """Default language to use for transcription."""

    default_response_format: ResponseFormat = ResponseFormat.TEXT
    """Default response format to use for transcription."""

    enable_ui: bool = True
    """Whether to enable the UI."""

    loopback_host_url: str | None = None
    """
    If set this is the URL that the gradio app will use to connect to the API server hosting speaches.
    If not set the gradio app will use the url that the user connects to the gradio app on.
    """

    moonshine: MoonshineConfig = MoonshineConfig()
    """Configuration for Moonshine ASR."""

    whisper: WhisperConfig = WhisperConfig()
    """Configuration for Whisper ASR."""

    # NOTE: options below are not used yet and should be ignored. Added as a placeholder for future features I'm currently working on.  # noqa: E501

    chat_completion_base_url: str = "https://api.openai.com/v1"
    chat_completion_api_key: str | None = None
    chat_completion_model: str | None = None

    speech_base_url: str | None = None
    speech_api_key: str | None = None
    speech_model: str = "piper"
    speech_extra_body: dict = {"sample_rate": 24000}

    transcription_base_url: str | None = None
    transcription_api_key: str | None = None

    max_no_data_seconds: float = 1.0
    """
    Max duration to wait for the next audio chunk before transcription is finilized and connection is closed.
    Used only for live transcription (WS /v1/audio/transcriptions).
    """
    min_duration: float = 1.0
    """
    Minimum duration of an audio chunk that will be transcribed.
    Used only for live transcription (WS /v1/audio/transcriptions).
    """
    word_timestamp_error_margin: float = 0.2
    """
    Used only for live transcription (WS /v1/audio/transcriptions).
    """
    max_inactivity_seconds: float = 2.5
    """
    Max allowed audio duration without any speech being detected before transcription is finilized and connection is closed.
    Used only for live transcription (WS /v1/audio/transcriptions).
    """  # noqa: E501
    inactivity_window_seconds: float = 5.0
    """
    Controls how many latest seconds of audio are being passed through VAD. Should be greater than `max_inactivity_seconds`.
    Used only for live transcription (WS /v1/audio/transcriptions).
    """  # noqa: E501
