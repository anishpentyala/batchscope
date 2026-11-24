"""Identity-preserving differences between two declared finite loading plans."""
from .core import InvalidPlan, analyze


def compare_plans(before, after):
    """Compare counts on an identical ID universe; order may deliberately change.

    Positive deltas are after minus before. No arrival ordering, optimizer or
    real-run equivalence is implied, even when all accounting is unchanged.
    """
    left, right = analyze(before), analyze(after)
    if set(before['ids']) != set(after['ids']):
        raise InvalidPlan('comparison requires the same declared ID universe')
    right_ids = {row['id']: row for row in right['ids']}
    rows = []
    for old in left['ids']:
        new = right_ids[old['id']]
        deltas = {key: new[key] - old[key]
                  for key in ('assigned', 'emitted', 'loader_dropped', 'sampler_removed')}
        if any(deltas.values()):
            rows.append({'id': old['id'], 'before': dict(old), 'after': dict(new),
                         'delta': deltas})
    ranks = []
    for rank in range(max(len(left['rank_batches']), len(right['rank_batches']))):
        old = left['rank_batches'][rank] if rank < len(left['rank_batches']) else None
        new = right['rank_batches'][rank] if rank < len(right['rank_batches']) else None
        ranks.append({'rank': rank, 'before': old, 'after': new,
                      'delta': None if old is None or new is None else new - old})
    totals = {key: {'before': left[key], 'after': right[key],
                    'delta': right[key] - left[key]}
              for key in ('assigned_occurrences', 'emitted_occurrences', 'distinct_emitted',
                          'missing_count', 'repeated_id_count', 'repeat_occurrences',
                          'sampler_removed_count', 'sampler_padding_count',
                          'loader_dropped_occurrences')}
    old_consumers = {(c['rank'], c['worker']): c for c in left['consumers']}
    new_consumers = {(c['rank'], c['worker']): c for c in right['consumers']}
    # Different models cannot be called equivalent merely because their totals agree.
    changed = (bool(rows) or left['model'] != right['model']
               or old_consumers != new_consumers or left['warnings'] != right['warnings']
               or any(value['delta'] for value in totals.values()))
    return {'schema': 'batchscope-comparison-v1',
            'status': 'CHANGED_ACCOUNTING' if changed else 'SAME_ACCOUNTING',
            'universe_size': left['universe_size'],
            'input_order_changed': before['ids'] != after['ids'],
            'before': {'model': left['model'], 'status': left['status'],
                       'warnings': left['warnings']},
            'after': {'model': right['model'], 'status': right['status'],
                      'warnings': right['warnings']},
            'totals': totals, 'rank_batches': ranks,
            'changed_id_count': len(rows), 'ids': rows,
            'emitted_gain_occurrences': sum(max(0, row['delta']['emitted']) for row in rows),
            'emitted_loss_occurrences': sum(max(0, -row['delta']['emitted']) for row in rows),
            'consumer_counts_changed': old_consumers != new_consumers,
            'scope': ('declared finite accounting only; same accounting does not establish '
                      'same ordering, consumer identity ownership, synchronization or real consumption')}
