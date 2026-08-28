# Raspberry Pi 5 latency and energy protocol

Use batch size one, `torch.inference_mode()`, fixed thread count and fixed software/model configuration. Warm up before timing; use thousands of timed inferences. Report median and p95 latency. Measure power with an external meter over a sufficiently long inference loop, and measure idle power under the same configuration.

Report both:

- system energy per inference from measured active energy divided by inference count;
- dynamic energy above idle using the measured idle baseline.

Do not infer energy from PSU rating or analytical MAC count.
