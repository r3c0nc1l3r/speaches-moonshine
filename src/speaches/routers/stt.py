import asyncio
from collections.abc import Generator, Iterable, AsyncGenerator
import logging
from typing import Annotated, Literal, Any

from fastapi import (
    APIRouter,
    Form,
    Request,
    Response,
    Depends,
)
from fastapi.responses import StreamingResponse
from faster_whisper.transcribe import BatchedInferencePipeline, TranscriptionInfo
from pydantic import Field
import numpy as np

from speaches.api_types import (
    DEFAULT_TIMESTAMP_GRANULARITIES,
    TIMESTAMP_GRANULARITIES_COMBINATIONS,
    CreateTranscriptionResponseJson,
    CreateTranscriptionResponseVerboseJson,
    TimestampGranularities,
    TranscriptionSegment,
)
from speaches.audio import AudioFileDependency
from speaches.config import Config
from speaches.dependencies import (
    ConfigDependency,
    WhisperModelManagerDependency,
    MoonshineModelManagerDependency,
)
from speaches.text_utils import segments_to_srt, segments_to_text, segments_to_vtt, get_timestamp_granularities

logger = logging.getLogger(__name__)

router = APIRouter(tags=["automatic-speech-recognition"])

type ResponseFormat = Literal["text", "json", "verbose_json", "srt", "vtt"]

# https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-response_format
DEFAULT_RESPONSE_FORMAT: ResponseFormat = "json"


def segments_to_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> Response:
    segments = list(segments)
    match response_format:
        case "text":
            return Response(segments_to_text(segments), media_type="text/plain")
        case "json":
            return Response(
                CreateTranscriptionResponseJson.from_segments(segments).model_dump_json(),
                media_type="application/json",
            )
        case "verbose_json":
            return Response(
                CreateTranscriptionResponseVerboseJson.from_segments(segments, transcription_info).model_dump_json(),
                media_type="application/json",
            )
        case "vtt":
            return Response(
                "".join(segments_to_vtt(segment, i) for i, segment in enumerate(segments)), media_type="text/vtt"
            )
        case "srt":
            return Response(
                "".join(segments_to_srt(segment, i) for i, segment in enumerate(segments)), media_type="text/plain"
            )


def format_as_sse(data: str) -> str:
    return f"data: {data}\n\n"


def segments_to_streaming_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> StreamingResponse:
    def segment_responses() -> Generator[str, None, None]:
        for i, segment in enumerate(segments):
            if response_format == "text":
                data = segment.text
            elif response_format == "json":
                data = CreateTranscriptionResponseJson.from_segments([segment]).model_dump_json()
            elif response_format == "verbose_json":
                data = CreateTranscriptionResponseVerboseJson.from_segment(
                    segment, transcription_info
                ).model_dump_json()
            elif response_format == "vtt":
                data = segments_to_vtt(segment, i)
            elif response_format == "srt":
                data = segments_to_srt(segment, i)
            yield format_as_sse(data)

    return StreamingResponse(segment_responses(), media_type="text/event-stream")


ModelId = Annotated[
    str,
    Field(
        description="The ID of the model. You can get a list of available models by calling `/v1/models`.",
        examples=[
            "Systran/faster-distil-whisper-large-v3",
            "bofenghuang/whisper-large-v2-cv11-french-ct2",
        ],
    ),
]


@router.post(
    "/v1/audio/translations",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
)
def translate_file(
    config: ConfigDependency,
    model_manager: ModelManagerDependency,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
    stream: Annotated[bool, Form()] = False,
    vad_filter: Annotated[bool, Form()] = False,
) -> Response | StreamingResponse:
    with model_manager.load_model(model) as whisper:
        whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
        segments, transcription_info = whisper_model.transcribe(
            audio,
            task="translate",
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=vad_filter,
        )
        segments = TranscriptionSegment.from_faster_whisper_segments(segments)

        if stream:
            return segments_to_streaming_response(segments, transcription_info, response_format)
        else:
            return segments_to_response(segments, transcription_info, response_format)


# HACK: Since Form() doesn't support `alias`, we need to use a workaround.
async def get_timestamp_granularities(request: Request) -> TimestampGranularities:
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return DEFAULT_TIMESTAMP_GRANULARITIES
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    assert timestamp_granularities in TIMESTAMP_GRANULARITIES_COMBINATIONS, (
        f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`."
    )
    return timestamp_granularities


# https://platform.openai.com/docs/api-reference/audio/createTranscription
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8915
@router.post(
    "/v1/audio/transcriptions",
    response_model=str | CreateTranscriptionResponseJson | CreateTranscriptionResponseVerboseJson,
)
def transcribe_file(
    config: ConfigDependency,
    whisper_manager: WhisperModelManagerDependency,
    moonshine_manager: MoonshineModelManagerDependency,
    request: Request,
    audio: AudioFileDependency,
    model: Annotated[ModelId, Form()],
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,
    timestamp_granularities: Annotated[
        TimestampGranularities,
        Form(alias="timestamp_granularities[]"),
    ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    vad_filter: Annotated[bool, Form()] = False,
) -> Response | StreamingResponse:
    timestamp_granularities = asyncio.run(get_timestamp_granularities(request))
    if timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES and response_format != "verbose_json":
        logger.warning(
            "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to `verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities."
        )

    # Check if the model is a Moonshine model
    if model.startswith("usefulsensors/moonshine"):
        with moonshine_manager.load_model(model) as moonshine:
            # Convert audio to numpy array
            audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Process with Moonshine
            # Note: Moonshine models don't support all the features that Whisper does
            # so we'll use basic transcription only
            transcription = moonshine.translate_audio(audio_array)
            
            # Format response
            if response_format == "text":
                return transcription
            elif response_format == "json":
                return {"text": transcription}
            else:  # verbose_json
                return {
                    "text": transcription,
                    "segments": [
                        {
                            "text": transcription,
                            "start": 0.0,
                            "end": len(audio_array) / 16000.0,  # Assuming 16kHz audio
                            "words": []  # Moonshine doesn't provide word-level timestamps
                        }
                    ],
                    "language": language or "en"
                }
    else:
        # Original Whisper model handling
        with whisper_manager.load_model(model) as whisper:
            whisper_model = BatchedInferencePipeline(model=whisper) if config.whisper.use_batched_mode else whisper
            segments, transcription_info = whisper_model.transcribe(
                audio,
                task="transcribe",
                language=language,
                initial_prompt=prompt,
                word_timestamps="word" in timestamp_granularities,
                temperature=temperature,
                vad_filter=vad_filter,
                hotwords=hotwords,
            )
            segments = TranscriptionSegment.from_faster_whisper_segments(segments)

            if stream:
                return segments_to_streaming_response(segments, transcription_info, response_format)
            else:
                return segments_to_response(segments, transcription_info, response_format)
