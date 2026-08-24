# Raspberry Pi 5 latency and energy protocol

Recommended reporting configuration:

1. Raspberry Pi 5, CPU inference, batch size 1.
2. Record OS, PyTorch version, CPU governor, thread count and ambient conditions.
3. Use `scripts/edge_benchmark.py` for warm-up plus median / mean / p95 latency.
4. Measure idle input power with the same peripherals and software state.
5. Run `scripts/power_benchmark_loop.py` for 60 s while an external meter logs
   timestamped input power.
6. Use `scripts/energy_from_powerlog.py` to integrate measured power over exactly the
   inference-loop interval.

Report:

- `E_system = integral(P_active(t) dt) / N_inferences`
- `E_dynamic = [integral(P_active(t) dt) - P_idle * T] / N_inferences`

Do not derive energy from the Raspberry Pi PSU rating or TDP.
