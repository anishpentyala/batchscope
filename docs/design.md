# Model, not observation

The only inputs are an immutable caller-supplied ID order and explicit arithmetic
rules. A passing report is conditional on that model. No framework config is
inferred and no caller code is executed. This is distinct from auditing observed
loader trajectories. The output deliberately includes every ID, not just a few
examples, so coverage numbers can be reconciled.

Map path: global truncated/padded index stream, rank stride, then rank-local
batch truncation. Iterable strided: combined rank-major consumer stride, then
per-consumer batch truncation. Iterable replicated: same full stream in every
consumer, then per-consumer batch truncation. Each emitted occurrence appears in
exactly one consumer; occurrence counts must reconcile before and after tails.

Resource admission: cap source bytes, IDs, consumer count and replicated assigned
occurrences before constructing queues. Reports still scale with IDs and hold
all counts in memory. No timing or RSS bound is promised.

Framework caveat: prospective identity coverage does not capture the order in
which worker batches arrive, actual optimizer steps, collective calls, filtering,
collation, training early exit or prefetch consumption. Unequal batch count is a
warning only. There is no assertion that a real DDP run hangs.

Research source ledger and rejected alternatives live in the private review
packet. TrainParity covers observed-ID audits, PyTorch documents the rules,
WebDataset supplies actual pipelines. No claim that the arithmetic itself is new.
