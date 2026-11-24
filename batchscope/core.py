"""Finite, explicitly ordered loading plans, not live loader inspection."""
from collections import Counter

MAX_IDS = 100_000
MAX_CONSUMERS = 1024
MAX_ASSIGNMENTS = 1_000_000
MAX_TRACE_IDS = 64


class InvalidPlan(ValueError):
    pass


def integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise InvalidPlan(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def analyze(plan):
    return _analyze(plan)


def _analyze(plan, emission_counts=None, emission_sequences=None):
    if type(plan) is not dict:
        raise InvalidPlan('plan must be an object')
    required = {'ids', 'mode', 'ranks', 'workers', 'batch_size', 'loader_drop_last'}
    allowed = required | {'sampler_drop_last', 'trace_ids'}
    if set(plan) - allowed or required - set(plan):
        raise InvalidPlan('unknown or missing plan fields')
    ids = plan['ids']
    if type(ids) is not list or len(ids) > MAX_IDS:
        raise InvalidPlan(f'ids must be an array of at most {MAX_IDS} values')
    if any(type(x) is not str or not x or len(x) > 256 for x in ids):
        raise InvalidPlan('each ID must be a nonempty string of at most 256 characters')
    if len(set(ids)) != len(ids):
        raise InvalidPlan('IDs must be unique in the declared universe')
    trace_ids = plan.get('trace_ids', [])
    if type(trace_ids) is not list or len(trace_ids) > MAX_TRACE_IDS:
        raise InvalidPlan(f'trace_ids must be an array of at most {MAX_TRACE_IDS} IDs')
    universe = set(ids)
    if any(type(x) is not str or x not in universe for x in trace_ids):
        raise InvalidPlan('trace_ids must contain only declared IDs')
    if len(set(trace_ids)) != len(trace_ids):
        raise InvalidPlan('trace_ids must be unique')
    trace_rows = {x: [] for x in trace_ids}
    mode = plan['mode']
    if type(mode) is not str or mode not in ('distributed_map', 'iterable_strided', 'iterable_replicated'):
        raise InvalidPlan('unsupported mode')
    ranks = integer(plan['ranks'], 'ranks', 1, MAX_CONSUMERS)
    workers = integer(plan['workers'], 'workers', 1, MAX_CONSUMERS)
    if ranks * workers > MAX_CONSUMERS:
        raise InvalidPlan(f'at most {MAX_CONSUMERS} rank/worker consumers')
    batch = integer(plan['batch_size'], 'batch_size', 1, MAX_ASSIGNMENTS)
    drop = plan['loader_drop_last']
    if type(drop) is not bool:
        raise InvalidPlan('loader_drop_last must be boolean')
    if mode == 'distributed_map':
        if type(plan.get('sampler_drop_last')) is not bool:
            raise InvalidPlan('distributed_map requires boolean sampler_drop_last')
    elif 'sampler_drop_last' in plan:
        raise InvalidPlan('sampler_drop_last is only valid for distributed_map')
    n = len(ids)
    if mode == 'iterable_replicated':
        size = n * ranks * workers
    elif mode == 'distributed_map' and not plan['sampler_drop_last']:
        size = ((n + ranks - 1) // ranks) * ranks
    else:
        size = n
    if size > MAX_ASSIGNMENTS:
        raise InvalidPlan(f'plan exceeds {MAX_ASSIGNMENTS} assigned occurrences')
    sampler_removed = []
    padding = []
    consumers = []
    assigned = Counter()
    emitted = Counter()
    discarded = Counter()
    rank_steps = [0] * ranks
    if mode == 'distributed_map':
        if plan['sampler_drop_last']:
            total = (n // ranks) * ranks
            scheduled = list(ids[:total])
            sampler_removed = list(ids[total:])
        else:
            total = ((n + ranks - 1) // ranks) * ranks
            padding = [ids[i % n] for i in range(total - n)] if n else []
            scheduled = ids + padding
        groups = [(rank, None, scheduled[rank::ranks]) for rank in range(ranks)]
    else:
        groups = []
        for rank in range(ranks):
            for worker in range(workers):
                offset = rank * workers + worker
                group = ids[offset::ranks * workers] if mode == 'iterable_strided' else list(ids)
                groups.append((rank, worker, group))
    for rank, worker, group in groups:
        keep = len(group) - len(group) % batch if drop else len(group)
        active, tail = group[:keep], group[keep:]
        steps = (keep + batch - 1) // batch
        rank_steps[rank] += steps
        if emission_sequences is not None:
            emission_sequences[rank, worker] = active
        if emission_counts is not None:
            emission_counts.update((identity, rank, worker) for identity in active)
        assigned.update(group)
        emitted.update(active)
        discarded.update(tail)
        if trace_ids:
            local_assigned = Counter(x for x in group if x in trace_rows)
            local_emitted = Counter(x for x in active if x in trace_rows)
            for x, count in local_assigned.items():
                trace_rows[x].append({'rank': rank, 'worker': worker,
                                      'assigned': count, 'emitted': local_emitted[x],
                                      'loader_dropped': count - local_emitted[x]})
        consumers.append({'rank': rank, 'worker': worker, 'assigned': len(group),
                          'emitted': keep, 'dropped_tail': len(tail), 'batches': steps,
                          'last_batch_size': (keep - (steps - 1) * batch) if steps else 0})
    removed_set = set(sampler_removed)
    rows = [{'id': x, 'assigned': assigned[x], 'emitted': emitted[x],
             'loader_dropped': discarded[x], 'sampler_removed': int(x in removed_set)}
            for x in ids]
    missing = [x for x in rows if x['emitted'] == 0]
    repeated = [x for x in rows if x['emitted'] > 1]
    exact = not missing and not repeated
    warnings = []
    if not ids:
        warnings.append('EMPTY_UNIVERSE')
    if len(set(rank_steps)) > 1:
        warnings.append('UNEQUAL_RANK_BATCH_COUNTS')
    if mode == 'distributed_map' and workers > 1:
        warnings.append('MAP_WORKERS_DO_NOT_REPARTITION_SAMPLER')
    expected_assigned = size - len(sampler_removed) if mode == 'distributed_map' and plan['sampler_drop_last'] else size
    assert sum(assigned.values()) == expected_assigned
    assert all(x['assigned'] == x['emitted'] + x['loader_dropped'] for x in rows)
    report = {'schema': 'batchscope-report-v1', 'model': mode,
            'status': 'EMPTY_UNIVERSE' if not ids else ('EXACTLY_ONCE' if exact else 'COVERAGE_CHANGED'),
            'universe_size': n, 'assigned_occurrences': sum(assigned.values()),
            'emitted_occurrences': sum(emitted.values()),
            'distinct_emitted': len(emitted),
            'missing_count': len(missing), 'repeated_id_count': len(repeated),
            'repeat_occurrences': sum(x['emitted'] - 1 for x in repeated),
            'sampler_removed_count': len(sampler_removed), 'sampler_padding_count': len(padding),
            'loader_dropped_occurrences': sum(discarded.values()),
            'rank_batches': rank_steps, 'warnings': warnings,
            'consumers': consumers, 'ids': rows}

    if 'trace_ids' in plan:
        report['traces'] = [
            {'id': x, 'sampler_removed': int(x in removed_set),
             'consumers': trace_rows[x]} for x in trace_ids]
    return report
