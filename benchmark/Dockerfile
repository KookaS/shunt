# Reproducible benchmark image — runs the caching/orchestration loop identically
# on any machine. Code is baked in; mount results.csv + reports read-write (see
# docker-compose.yml `benchmark` service) to persist regenerated data and plots.
#
# Build from the REPO ROOT so the `benchmark` package is importable:
#   docker build -f benchmark/Dockerfile -t shunt-benchmark .
# (BuildKit picks up benchmark/Dockerfile.dockerignore, overriding the root
# .dockerignore which excludes benchmark/ from the product image.)
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src:/app"

# Light runtime: only what the simulated loop + plot regeneration need. Embedding
# strategies (fastembed/hnswlib) are refreshed on the host, not in this image, so
# the container stays reproducible and offline.
RUN pip install --no-cache-dir "pyyaml>=6.0" "numpy>=1.26" "matplotlib>=3.9"

COPY src/ ./src/
COPY benchmark/ ./benchmark/
COPY pyproject.toml ./

# Simulated by default: reports missing/stale cells + regenerates plots, never
# fabricates outcomes. `--strategy full` = the exhaustive matrix loop (the runner's
# default is cost_optimal, which does not regenerate plots). Pass --live (with keys +
# Docker) for real execution.
ENTRYPOINT ["python", "-m", "benchmark.runner.run_matrix"]
CMD ["--strategy", "full", "--no-summary"]
