# batchscope

Inspect a finite data-loading plan before starting a training or evaluation run.
Supply an ordered list of unique sample IDs, explicit partition rules, batch size
and tail policies. See which identities would repeat or disappear under that
model, and how many local batches each rank would produce.

This is an experimental tool. It has no observed users. It does not
inspect or run a real DataLoader, read sample contents, launch training, or prove
that a configured training job follows this plan.

## A small but consequential tail

Five samples split across two ranks need six sampler slots when padding is on.
Then a loader that drops incomplete local batches can remove two occurrences,
including the extra padding. Counting padding alone misses the final coverage.

```sh
python -m pip install .
batchscope examples/padded.json
```

That example emits six occurrences of five identities, so it exits 1 with
`COVERAGE_CHANGED`. Change `loader_drop_last` to `true` and it emits four
identities once, omitting the fifth. This is invented input, not a measured job.

## Explicit models

| mode | ownership before batching | tail policy |
| --- | --- | --- |
| `distributed_map` | pad/truncate the supplied global order to a rank multiple, then stride by rank | loader tail is per rank; workers do not repartition sampler indices |
| `iterable_strided` | rank-major combined consumer index `rank * workers + worker`, stride by `ranks * workers` | one tail per rank/worker |
| `iterable_replicated` | every rank/worker sees the entire ordered universe | one tail per rank/worker |

`workers` means the number of independently iterated sources per rank. Use 1 for
single-process iterable loading. It is not a literal translation of PyTorch's
`num_workers=0`. Map-style worker count does not multiply or repartition samples.

These are declared mathematical models. The map model follows the documented
unshuffled `DistributedSampler` padding/truncation rule and ordinary local
batching. The iterable rules describe two explicitly chosen source strategies,
not arbitrary PyTorch iterable datasets. A different worker ordering or nested
sharding can produce different identities. Supply the actual intended order.

No RNG or shuffle emulation, custom sampler, dynamic filtering, resampling,
variable batch sizes, accumulation, retries, early exit, persistent-worker state,
resume state, sample packing, or automatic framework-config parsing. Do not use
these counts as evidence of optimizer updates or synchronized collective calls.

## Input

```json
{
  "ids": ["a", "b", "c", "d", "e"],
  "mode": "distributed_map",
  "ranks": 2,
  "workers": 1,
  "batch_size": 2,
  "sampler_drop_last": false,
  "loader_drop_last": false
}
```

All fields shown are required for `distributed_map`. The other modes require all
except `sampler_drop_last`, which is rejected rather than silently ignored.
IDs are nonempty unique strings. Distinct IDs can contain identical sample
contents; this tool cannot detect that. No whitespace or Unicode normalization.
Unknown fields, duplicate JSON members, NaN/Infinity, invalid UTF-8, nonregular
input files and plans over 8 MiB are rejected. Limits: 100,000 IDs, 1,024 combined
consumers, 1,000,000 assigned occurrences. These are admission limits, not a hard
RAM/time guarantee. A regular file on a slow filesystem can still block.

## Selected identity witnesses

Add optional `"trace_ids": ["a", "e"]` to the input to locate selected IDs within
this finite model. Select at most 64 distinct IDs from the declared universe.
The report adds `traces`, in selection order. Each trace contains the ID,
`sampler_removed` (0 or 1), and a rank/worker-ordered `consumers` list containing
only consumers assigned that ID. Each consumer has `assigned`, `emitted`, and
`loader_dropped` occurrence counts. Map-style consumers have `worker: null`;
map workers do not define independent sample ownership here.

For the example above, tracing `a` shows one assigned occurrence on each rank.
With `loader_drop_last: true`, rank 0 retains its occurrence and rank 1 drops its
occurrence. An ID removed by global sampler truncation has `sampler_removed: 1`
and an empty consumer list. An empty selection emits `traces: []`; omitting
`trace_ids` preserves the original report shape. Traces describe declared
ownership and tail arithmetic, not arrival order, observed consumption, or
whether identical contents occur under other IDs. Reports still disclose IDs.

## Audit explicit observed receipts

```sh
batchscope --receipts plan.json receipts.json
```

Receipts are a JSON array of individual retained occurrences, for example
`[{"id": "a", "rank": 0, "worker": null}]`. Each object requires exactly these
three fields. Record at the boundary you intend to audit, and supply a complete
list for the declared finite run. Nothing here records or authenticates events.
Repeated entries are separate occurrences, not deduplicated records. Unknown
nonempty IDs are reported as unexpected evidence, not silently discarded.
Rank must exist in the plan. Iterable worker must exist too. Map worker must be
`null`, because this model does not predict map-worker identity ownership.

The `batchscope-receipts-v1` report compares occurrence counts per exact
`(id, rank, worker)` key. It reports every differing key, missing and extra
occurrences separately, and the number of unknown-ID occurrences (a subset of
extras). A consumer swap therefore produces a missing and an extra occurrence
even when global totals match. Known rows follow plan ID order and then consumer
order; unknown IDs follow first-observed ID order. Arrival order is not checked.
The full baseline coverage status and warnings remain visible as separate fields.

Exit 0 means `MATCHED_RECEIPTS`, exit 1 means `RECEIPT_MISMATCH`, and exit 2 means
invalid input or failed output delivery. Matching receipts does **not** mean
exactly-once coverage: they can match a plan that deliberately duplicates or
omits IDs, or an empty plan. It also cannot establish receipt completeness,
provenance, real consumption, sample contents, batch boundaries, execution order,
optimizer updates or synchronized collective calls. An incomplete receipt export
will look like missing occurrences. Use the retained evidence and its collection
boundary to interpret a mismatch, not as a diagnosis of its cause.

The receipt file uses the same strict JSON and 8 MiB file limits as a plan, and
at most 1,000,000 occurrence objects. IDs use the plan's 256-character limit.
Receipts may disclose sensitive IDs. The API is `audit_receipts(plan, receipts)`;
inputs are not mutated. Single-plan and comparison outputs are unchanged.

## Check order within each modeled consumer

```sh
batchscope --ordered-receipts plan.json receipts.json
```

This opt-in mode uses the same strict receipt contract and count differences as
`--receipts`, then also checks the order of IDs within each modeled consumer.
The API is `audit_receipts(plan, receipts, check_order=True)`. The default API
and `--receipts` reports retain their original shape and count-only semantics.

Input array order must reflect the collection boundary's order for each consumer.
Events from different consumers may interleave freely. Map consumers are ranks
with `worker: null`; iterable consumers are rank/worker pairs. This does not
compare a global arrival order, infer PyTorch's worker scheduling, or reconstruct
batches. Do not use this option on an unordered log export or a loader configured
to return items in a different order from the declared model.

The report uses `batchscope-ordered-receipts-v1`. `order_differences` lists each
mismatched consumer in rank/worker order, its expected and observed lengths, and
its first differing zero-based local index with `expected_id` and `observed_id`.
A missing suffix has `observed_id: null`; an extra suffix has `expected_id: null`.
Only the first local divergence is reported. These are positional witnesses,
not an edit script, causal diagnosis, or alignment after an insertion/deletion.
Repeated occurrences of the same ID are indistinguishable. It cannot detect a
swap between equal IDs or different sample contents carrying the same ID.

Exit 0 is `MATCHED_ORDERED_RECEIPTS`; exit 1 is `ORDERED_RECEIPT_MISMATCH`; exit 2
is invalid input or failed delivery. As with count-only audits, a match may agree
with a duplicating, incomplete or empty plan. The evidence remains conditional
on complete, correctly labeled receipts, not authenticated runtime consumption,
exactly-once coverage, batch boundaries or training behavior.

## Compare two explicit plans

```sh
batchscope --compare before.json after.json
```

Both inputs use the same validation and file limits as a single plan. They must
declare the same unique ID universe, but may reorder it or change any explicit
loading rule. No union, intersection, coercion or implicit ID alignment is used.
Rows align by exact ID and are reported in the before plan's order.

The `batchscope-comparison-v1` report contains before/after coverage statuses,
aggregate before/after counts and signed deltas (after minus before), per-rank
batch counts, and every ID whose assigned/emitted/loader-dropped/sampler-removed
accounting changes. Added or removed ranks use `null` for the absent side and
its delta, rather than pretending an absent rank had zero batches. Separate
`emitted_gain_occurrences` and `emitted_loss_occurrences` prevent equal aggregate
counts from concealing identity swaps. Changing input order can change which
IDs survive tails even when emitted totals and batch counts stay the same.

Comparison exits 0 for `SAME_ACCOUNTING`, 1 for `CHANGED_ACCOUNTING`, and 2 for
invalid inputs or failed output delivery. A model change, consumer-count change
or warning change counts as changed accounting too. `input_order_changed`
reports reordering separately; unchanged accounting is **not** a claim of equal
order, per-consumer identity ownership, statistical equivalence, synchronization
or actual consumption. In particular, two identically incomplete/duplicating
plans can return `SAME_ACCOUNTING`; inspect both embedded coverage statuses.
Optional trace selections do not affect comparison. Full single-plan reports
remain available separately; this comparison does not embed their trace rows.

Python API: `from batchscope import compare_plans`, then
`compare_plans(before_dict, after_dict)`. Inputs are not modified. All counts and
IDs describe declared finite models only. Comparisons are source-only and are
not part of any published package release.

## Report and exits

Every declared ID has assigned, emitted and loader-dropped multiplicities, plus
whether global sampler truncation removed it. Each consumer reports counts,
batches and final batch size; per-rank batches sum that rank's worker batches.

- 0: `EXACTLY_ONCE` under the supplied finite model.
- 1: `COVERAGE_CHANGED` or `EMPTY_UNIVERSE`. Review the counts.
- 2: invalid input, usage or output delivery failure. No usable complete report.

The full report must be written and flushed before exit 0 or 1. A broken output
may retain a JSON prefix, so use a report only with successful delivery and its
expected exit status. Redirecting stdout is not an atomic output transaction. A short `write` return
or a failed flush is an output failure too, even if no exception was raised by
`write`. The callable `batchscope.__main__.main()` returns 2 without replacing
or closing caller-owned stdout. The module and installed console use a separate
process entrypoint to prevent a failed buffered stdout from being retried during
interpreter shutdown. Help output follows the same complete-delivery contract.
These process controls are tested on Linux CPython; no Windows claim is made.

`UNEQUAL_RANK_BATCH_COUNTS` is a planning warning, not proof of a DDP hang.
Actual synchronization, masking and join handling are outside this tool.
`EXACTLY_ONCE` does not establish randomness, class balance, content uniqueness,
annotation quality, statistical comparability, or actual sample consumption.
Reports contain the supplied IDs. Review before sharing them.

## Why another tool?

[TrainParity](https://github.com/intelland/TrainParity) checks observed stable-ID
trajectories against a declared coverage policy and also tests actual training
parity. Use it to audit observed runs. This tool instead computes a prospective
finite plan without executing training code or importing a framework.

[PyTorch documents](https://docs.pytorch.org/docs/main/utils/data.html) sampler
padding/truncation and independent iterable-worker tails. Its `len(dataloader)`
estimate may miss fragmented tails. This tool does not fix those semantics; it
makes a limited, explicit plan inspectable with identity-level accounting.

[WebDataset](https://webdataset.github.io/webdataset/webdataset/) offers actual
sharding, batching and epoch controls. This tool does not emulate its
pipeline or resampling rules.

## Development

Python 3.10+, standard-library runtime. `python -m unittest discover -s tests -v`.
The tests include an independent event-by-event reference over 9,360 small plans,
regular input/output failures, empty sources, Unicode IDs and invalid inputs.
The [first hosted run](https://github.com/anishpentyala/batchscope/actions/runs/45344164628)
passed the standard-library and installed-wheel checks on Python 3.10, 3.13 and
3.15. Its Python 3.13 CPU job also passed three real PyTorch 2.15.0+cpu tests:
unshuffled map sampling with local batch tails, two-worker map loading, and the
explicit strided/replicated iterable sources with zero or two DataLoader workers.
Those controls compare emitted identity multiplicities and local batch counts,
not arrival order, optimizer updates or arbitrary pipelines. NumPy was not
installed in that CPU job; PyTorch warned about its unavailable NumPy bridge,
which these tests do not use. Other framework versions, GPUs and custom loading
strategies remain unvalidated. Implementation and tests were produced on Anish's behalf; no claim of his
personal code review or use is made.

## Audit explicit batch boundaries

```sh
batchscope --batch-receipts plan.json batches.json
```

A batch file is a JSON array of objects with exactly `rank`, `worker` and `ids`,
for example `[{"rank": 0, "worker": null, "ids": ["a", "b"]}]`.
Each `ids` is a nonempty list of retained occurrences. Repeated IDs remain
separate occurrences. Rank/worker rules are the same as receipts above. Empty
batches are rejected; an empty top-level array represents no retained batches.
At most1,000,000 batches and1,000,000 total occurrences are admitted, subject to
the same8MiB CLI input limit. Unknown nonempty IDs remain mismatch evidence.

The API is `audit_batches(plan, batches)`. This opt-in mode compares exact
local batch ID lists against consecutive `batch_size` chunks of each modeled
consumer's emitted sequence, including a permitted short final batch. Input
array order must preserve each consumer's local batch order. Different consumers
may interleave freely. For iterable models, consumers are rank/worker pairs,
not reconstructed rank-level loader batches. Map workers remain `null`.

The `batchscope-batch-receipts-v1` report embeds the full `ordered_receipts`
report, including planned coverage status, count and local order differences.
`boundary_differences` gives each mismatched consumer's expected/observed batch
counts and its first differing zero-based local batch index with full
`expected_ids`/`observed_ids`. Missing and extra batches use `null` on the absent
side. Only the first local difference is retained, not a sequence alignment or
causal diagnosis. Reports disclose IDs. Existing receipt modes are unchanged.

Thus `[a,b],[c]` differs from `[a],[b,c]` even though flattened ID order and counts
agree. Exit0 is `MATCHED_BATCH_RECEIPTS`, exit1 is `BATCH_RECEIPT_MISMATCH`, and
exit2 is invalid input or failed delivery. A match can agree with an empty,
omitting or duplicating plan. This does not authenticate collection completeness,
prove runtime consumption or contents, infer cross-consumer order, or establish
optimizer updates, synchronized collectives or exactly-once coverage.
