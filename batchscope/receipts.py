"""Compare caller-supplied ID receipts with a declared finite plan."""
from collections import Counter
from .core import InvalidPlan, MAX_ASSIGNMENTS, _analyze, integer


def audit_receipts(plan, receipts, *, check_order=False):
    """Audit an explicit complete receipt list, not the runtime or its provenance.

    Each occurrence has exactly id, rank, worker. Map workers are null because
    this model does not determine which map worker fetched a sample.
    """
    if type(check_order) is not bool:
        raise InvalidPlan('check_order must be boolean')
    expected = Counter()
    sequences = {} if check_order else None
    baseline = _analyze(plan, expected, sequences)
    positions = {}
    first_mismatch = {}
    if type(receipts) is not list or len(receipts) > MAX_ASSIGNMENTS:
        raise InvalidPlan(f'receipts must be an array of at most {MAX_ASSIGNMENTS} occurrences')
    observed = Counter()
    universe = set(plan['ids'])
    map_mode = plan['mode'] == 'distributed_map'
    for event in receipts:
        if type(event) is not dict or set(event) != {'id', 'rank', 'worker'}:
            raise InvalidPlan('each receipt must contain exactly id, rank, worker')
        identity = event['id']
        if type(identity) is not str or not identity or len(identity) > 256:
            raise InvalidPlan('receipt ID must be a nonempty string of at most 256 characters')
        rank = integer(event['rank'], 'receipt rank', 0, plan['ranks'] - 1)
        worker = event['worker']
        if map_mode:
            if worker is not None:
                raise InvalidPlan('map receipts require worker: null')
        else:
            integer(worker, 'receipt worker', 0, plan['workers'] - 1)
        observed[identity, rank, worker] += 1
        if check_order:
            consumer = rank, worker
            position = positions.get(consumer, 0)
            sequence = sequences[consumer]
            want = sequence[position] if position < len(sequence) else None
            if want != identity and consumer not in first_mismatch:
                first_mismatch[consumer] = {'index': position, 'expected_id': want,
                                            'observed_id': identity}
            positions[consumer] = position + 1
    # Preserve plan ID order, then unknown IDs in first-observed order. Consumer
    # sort is explicit so input arrival order never determines report order.
    identities = list(plan['ids']) + list(dict.fromkeys(
        event['id'] for event in receipts if event['id'] not in universe))
    by_id = {}
    for key in expected.keys() | observed.keys():
        by_id.setdefault(key[0], []).append(key)
    rows = []
    missing = extra = unknown = 0
    for identity in identities:
        for key in sorted(by_id.get(identity, []), key=lambda k: (k[1], -1 if k[2] is None else k[2])):
            want, got = expected[key], observed[key]
            missing += max(0, want - got)
            extra += max(0, got - want)
            if identity not in universe:
                unknown += got
            if want != got:
                rows.append({'id': identity, 'rank': key[1], 'worker': key[2],
                             'known_id': identity in universe,
                             'expected': want, 'observed': got, 'delta': got - want})
    report = {'schema': 'batchscope-receipts-v1',
            'status': 'MATCHED_RECEIPTS' if not rows else 'RECEIPT_MISMATCH',
            'model': plan['mode'], 'planned_coverage_status': baseline['status'],
            'planned_warnings': baseline['warnings'],
            'expected_occurrences': sum(expected.values()),
            'observed_occurrences': len(receipts),
            'missing_occurrences': missing, 'extra_occurrences': extra,
            'unknown_id_occurrences': unknown, 'differences': rows,
            'scope': 'Conditional on the supplied complete receipts and declared plan. '
                     'No provenance, runtime, order, batch-boundary, content, or exactly-once guarantee.'}

    if check_order:
        differences = []
        for (rank, worker), sequence in sequences.items():
            consumer = rank, worker
            count = positions.get(consumer, 0)
            mismatch = first_mismatch.get(consumer)
            if mismatch is None and count < len(sequence):
                mismatch = {'index': count, 'expected_id': sequence[count],
                            'observed_id': None}
            if mismatch is not None:
                differences.append({'rank': rank, 'worker': worker,
                                    'expected_length': len(sequence),
                                    'observed_length': count,
                                    'first_difference': mismatch})
        report['schema'] = 'batchscope-ordered-receipts-v1'
        report['status'] = ('MATCHED_ORDERED_RECEIPTS' if not rows and not differences
                            else 'ORDERED_RECEIPT_MISMATCH')
        report['order_differences'] = differences
        report['scope'] = ('Conditional on complete supplied receipts in each modeled consumer order. '
                           'No cross-consumer order, provenance, runtime, batch-boundary, content, '
                           'or exactly-once guarantee.')
    return report
