# CPU image. Build: docker build -t clipper .
# Run:   docker run --rm --env-file .env -v "$PWD/media:/media" -v clipper-home:/data \
#          clipper run /media/episode.mp4 --i-own-rights
FROM python:3.11-slim-bookworm

# Debian's ffmpeg is built with libass, libfribidi and libharfbuzz (needed for captions and RTL text).
# libgl1, libglib2.0-0, libportaudio2: loaded by MediaPipe's dependencies (OpenCV, sounddevice) for
# the face-following crop. Not yet verified in a Docker build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fontconfig libgl1 libglib2.0-0 libportaudio2 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra reframe --no-install-project
COPY src ./src
COPY assets ./assets
RUN uv sync --frozen --no-dev --extra reframe

ENV PATH=/opt/venv/bin:$PATH CLIPPER_HOME=/data
VOLUME ["/data"]
ENTRYPOINT ["clipper"]
CMD ["--help"]
