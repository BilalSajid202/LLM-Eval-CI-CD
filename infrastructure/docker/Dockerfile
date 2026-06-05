FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY data ./data
COPY prompts ./prompts
COPY migrations ./migrations

RUN pip install --no-cache-dir -e ".[scorers]"

ENV EVAL_MODE=local
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["llm-eval"]
CMD ["run", "--scope", "smoke"]
