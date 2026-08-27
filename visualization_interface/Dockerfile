FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/app:/opt/orbital_propagator/src

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m pip install --no-cache-dir "dvc[s3]==3.67.1"

COPY app ./app

ARG HOST_UID=1000
ARG HOST_GID=1000

RUN set -eux; \
    if getent group "${HOST_GID}" >/dev/null; then \
        GROUP_NAME="$(getent group "${HOST_GID}" | cut -d: -f1)"; \
    else \
        groupadd -g "${HOST_GID}" dvcgroup; \
        GROUP_NAME=dvcgroup; \
    fi; \
    useradd -m -u "${HOST_UID}" -g "${GROUP_NAME}" dvcuser

CMD ["python", "app/app.py"]
