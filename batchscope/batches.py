"""Compare explicit local batch boundaries with a declared finite plan."""
from .core import InvalidPlan, MAX_ASSIGNMENTS, _analyze, integer
from .receipts import audit_receipts


def audit_batches(plan, batches):
    """Audit supplied complete batches in each consumer's local order.

    Cross-consumer interleaving is ignored. Batches must be nonempty. Flattened
    ordered receipt evidence is retained separately, not confused with boundary
    agreement or proof of actual training behavior.
    """
    sequences = {}
    _analyze(plan, emission_sequences=sequences)
    if type(batches) is not list or len(batches) > MAX_ASSIGNMENTS:
        raise InvalidPlan(f'batches must be an array of at most {MAX_ASSIGNMENTS} batches')
    observed = {consumer: [] for consumer in sequences}
    receipts = []
    for batch in batches:
        if type(batch) is not dict or set(batch) != {'ids', 'rank', 'worker'}:
            raise InvalidPlan('each batch must contain exactly ids, rank, worker')
        rank = integer(batch['rank'], 'batch rank', 0, plan['ranks'] - 1)
        worker = batch['worker']
        if plan['mode'] == 'distributed_map':
            if worker is not None:
                raise InvalidPlan('map batches require worker: null')
        else:
            integer(worker, 'batch worker', 0, plan['workers'] - 1)
        ids = batch['ids']
        if type(ids) is not list or not ids:
            raise InvalidPlan('batch ids must be a nonempty array')
        if len(receipts) + len(ids) > MAX_ASSIGNMENTS:
            raise InvalidPlan(f'batches exceed {MAX_ASSIGNMENTS} total occurrences')
        for identity in ids:
            if type(identity) is not str or not identity or len(identity) > 256:
                raise InvalidPlan('batch ID must be a nonempty string of at most 256 characters')
            receipts.append({'id': identity, 'rank': rank, 'worker': worker})
        observed[rank, worker].append(ids)
    ordered = audit_receipts(plan, receipts, check_order=True)
    differences = []
    expected_total = 0
    size = plan['batch_size']
    for (rank, worker), sequence in sequences.items():
        expected = [sequence[i:i + size] for i in range(0, len(sequence), size)]
        got = observed[rank, worker]
        expected_total += len(expected)
        first = next((i for i in range(max(len(expected), len(got)))
                      if (expected[i] if i < len(expected) else None) !=
                         (got[i] if i < len(got) else None)), None)
        if first is not None:
            differences.append({'rank': rank, 'worker': worker,
                                'expected_batches': len(expected), 'observed_batches': len(got),
                                'first_difference': {
                                    'index': first,
                                    'expected_ids': expected[first] if first < len(expected) else None,
                                    'observed_ids': list(got[first]) if first < len(got) else None}})
    return {
        'schema': 'batchscope-batch-receipts-v1',
        'status': ('MATCHED_BATCH_RECEIPTS' if not differences and
                   ordered['status'] == 'MATCHED_ORDERED_RECEIPTS' else 'BATCH_RECEIPT_MISMATCH'),
        'expected_batches': expected_total, 'observed_batches': len(batches),
        'boundary_differences': differences, 'ordered_receipts': ordered,
        'scope': 'Conditional on complete supplied nonempty batches in each modeled consumer order. '
                 'No global order, provenance, runtime consumption, contents, optimizer updates, '
                 'synchronized collectives or exactly-once guarantee.',
    }
