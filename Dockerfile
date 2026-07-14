FROM python:3.12-slim AS builder

WORKDIR /build
COPY --link uv.lock pyproject.toml ./

RUN pip install --no-cache-dir uv \
 && uv export --no-dev --no-hashes --output-file=requirements.txt

FROM python:3.12-slim

WORKDIR /app
COPY --link --from=builder /build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --link src/ src/ pyproject.toml ./
RUN pip install --no-cache-dir .

EXPOSE 8080

ENV SHUNT_HOST=0.0.0.0
ENV SHUNT_PORT=8080

ENTRYPOINT ["python", "-m", "shunt"]
