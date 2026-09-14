# Screenshots

`grafana-l4-session.png` — the Grafana dashboard covering the Phase 6 GPU
sessions, referenced from the top-level README.

To reproduce: with the local compose stack up and Prometheus holding the
session data, open http://localhost:3000, set **Instance** to `gpu-pod`, set the
time range to the session window, and screenshot the full dashboard.
