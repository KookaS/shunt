FROM python:3.12-slim AS builder

# build-essential compiles the deps that ship no cp312 wheel (e.g. hnswlib, a C++
# extension). Kept in the builder ONLY — the runtime image copies the finished
# venv, so no compiler bloats the shipped image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# hnswlib's setup.py defaults to `-march=native`, which bakes the BUILD host's CPU
# instructions (e.g. AVX-512) into the wheel — a binary that SIGILLs (exit 132) on
# any machine whose CPU lacks them. GitHub's runner fleet is heterogeneous, so a
# `-march=native` build crashes on the legs that land on older CPUs. HNSWLIB_NO_NATIVE
# drops that flag (keeps `-O3`), yielding a portable x86-64-baseline binary.
ENV HNSWLIB_NO_NATIVE=1

WORKDIR /build
COPY --link uv.lock pyproject.toml ./

# --no-emit-project: export third-party deps only. The project itself is a `-e .`
# self-reference that `pip install -r` would try to build before src/ is copied
# (it fails: "does not appear to be a Python project"). The package is installed
# into the venv separately below, after its source lands.
RUN pip install --no-cache-dir uv \
 && uv export --no-dev --no-hashes --no-emit-project --output-file=requirements.txt

COPY --link src/ src/
RUN python -m venv /venv \
 && /venv/bin/pip install --no-cache-dir -r requirements.txt \
 && /venv/bin/pip install --no-cache-dir . \
 # Portability guard (structural check, not a note): a `-march=native` wheel bakes
 # AVX-512 into the binary and SIGILLs (exit 132) on any runner whose CPU lacks it.
 # objdump ships with build-essential's binutils; fail the build if an AVX-512 (zmm)
 # opcode slipped into the compiled hnswlib extension despite HNSWLIB_NO_NATIVE.
 && ! objdump -d /venv/lib/python3.12/site-packages/hnswlib*.so \
      | grep -qiE '%zmm|avx512'

FROM python:3.12-slim

WORKDIR /app
COPY --link --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

EXPOSE 8080

ENV SHUNT_HOST=0.0.0.0
ENV SHUNT_PORT=8080

ENTRYPOINT ["python", "-m", "shunt"]
