# Vulture false-positive whitelist (advisory dead-code scan, --min-confidence 80).
# Each name below is "used" here so vulture treats it as live. Reviewed entries:
#
# API-mandated signatures (must keep — the caller dictates the parameter list):
frame  # signal handler (sig, frame) required by signal.signal — src/shunt/proxy/server.py
sig  # signal handler param required by signal.signal — src/shunt/proxy/server.py
