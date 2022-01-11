FROM python:3.10-slim as builder

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /akd-app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install --without dev --no-root

FROM python:3.10-slim as runtime

ENV VIRTUAL_ENV=/akd-app/.venv \
    PATH="/akd-app/.venv/bin:$PATH"

WORKDIR /akd-app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY src/ src/
COPY app/ app/
COPY wsgi.py wsgi.py

ENV PYTHONPATH=/akd-app/src

EXPOSE 8070

CMD ["/akd-app/.venv/bin/uvicorn", "app.serve:app", "--host", "0.0.0.0", "--port", "8000"]
