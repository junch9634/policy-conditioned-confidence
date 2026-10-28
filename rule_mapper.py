import hashlib
import itertools
import json
STATES = {"TRUE", "FALSE", "UNDETERMINABLE", "UNREVIEWED"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def concrete(table, policy, world):
    for rule in sorted(policy['rules'], key=lambda r: r['priority']):
        if all(world[key] == value for key, value in rule['when'].items()):
            return rule['label'], rule['clause_id'], rule['priority']
    raise ValueError('POLICY_DEFINITION_UNRESOLVED: uncovered rule assignment')


def evaluate(table, record, policy_id):
    policy = table['policies'][policy_id]
    keys = table['decision_fact_ids']
    facts = record['facts']
    result = {'image_id': record['image_id'], 'policy_id': policy_id,
              'shared_fact_record_id': record['fact_record_id'],
              'shared_fact_revision': record['revision'], 'facts_sha256': fingerprint(record),
              'policy_sha256': policy['policy_sha256'],
              'source_type': 'HUMAN_FACT_RULE_DERIVED', 'human_validated': False,
              'rule_table_human_validated': False, 'rule_derived_label': None,
              'status': 'ADDITIONAL_OBSERVATION_REQUIRED', 'applied_clauses': [],
              'supporting_observations': [], 'needed_fact_ids': [],
              'needs_supported_possibilities': False, 'unresolved_issues': [],
              'inference_basis': None, 'possibility_outcomes': [],
              'not_y_gold': True}
    if record.get('definition_issue'):
        result.update(status='POLICY_DEFINITION_UNRESOLVED',
                      unresolved_issues=[record['definition_issue']])
        return result
    if any(facts[k]['state'] not in STATES for k in keys):
        raise ValueError('Invalid observation state')
    for key in keys:
        if facts[key]['state'] != 'UNREVIEWED' and not facts[key].get('sources'):
            raise ValueError('Observed fact lacks provenance: ' + key)
    possibilities = []
    for values in itertools.product([True, False], repeat=len(keys)):
        world = dict(zip(keys, values))
        if all(facts[k]['state'] not in {'TRUE', 'FALSE'} or
               world[k] == (facts[k]['state'] == 'TRUE') for k in keys):
            label, clause, priority = concrete(table, policy, world)
            possibilities.append({'world': world, 'label': label, 'clause_id': clause, 'priority': priority})
    labels = {w['label'] for w in possibilities}
    observed = [{'fact_id': key, **facts[key]} for key in keys if facts[key]['state'] in {'TRUE', 'FALSE'}]

    def finish(label, worlds, basis, support):
        result.update(rule_derived_label=label, status='RULE_DERIVED_COMPLETE',
                      inference_basis=basis, supporting_observations=support)
        seen = set()
        for world in worlds:
            if world['clause_id'] not in seen:
                seen.add(world['clause_id'])
                result['applied_clauses'].append({'clause_id': world['clause_id'], 'priority': world['priority'],
                    'verbatim': table['clauses'][world['clause_id']]['verbatim']})
        return result

    # These completions are a logical short-circuit proof, not asserted visual possibilities.
    if len(labels) == 1:
        return finish(next(iter(labels)), possibilities, 'SAME_RESULT_FOR_ALL_UNCHECKED_BRANCHES', observed)

    influential = []
    for key in keys:
        if facts[key]['state'] in {'TRUE', 'FALSE'}:
            continue
        for a in possibilities:
            if any(a['label'] != b['label'] and all(a['world'][other] == b['world'][other]
                    for other in keys if other != key) for b in possibilities):
                influential.append(key)
                break
    result['needed_fact_ids'] = [k for k in influential if facts[k]['state'] == 'UNREVIEWED']
    if result['needed_fact_ids']:
        result['unresolved_issues'] = ['Relevant facts have not been reviewed; no UNKNOWN label is inferred.']
        return result

    evidence = record['possibilities']
    result['needs_supported_possibilities'] = True
    if evidence['status'] != 'SUPPORTED_SET_RECORDED':
        result['unresolved_issues'] = ['Observed uncertainty alone does not identify a complete, evidence-supported possibility set.']
        return result
    if not evidence.get('complete_set_confirmed') or not evidence.get('sources') or not evidence['branches']:
        result['unresolved_issues'] = ['A complete supported possibility set and provenance are missing.']
        return result
    supported = []
    for branch in evidence['branches']:
        definition = table['possibility_branches'][branch['branch_id']]
        worlds = [w for w in possibilities if all(w['world'][k] == v for k, v in definition['when'].items())]
        if not worlds or not branch.get('rationale', '').strip():
            result.update(status='HUMAN_FACT_CONFLICT', unresolved_issues=['Possibility conflicts with a confirmed fact or lacks evidence.'])
            return result
        supported.extend(worlds)
        result['possibility_outcomes'].append({'branch_id': branch['branch_id'],
            'rationale': branch['rationale'], 'outcomes': sorted({w['label'] for w in worlds})})
    final = {w['label'] for w in supported}
    label = next(iter(final)) if len(final) == 1 else 'UNKNOWN'
    result['needs_supported_possibilities'] = False
    result['unresolved_issues'] = []
    result = finish(label, supported, 'HUMAN_SUPPORTED_POSSIBILITIES_UNDER_ORIGINAL_UNCERTAINTY_RULE',
                    [{'fact_id': k, **facts[k]} for k in keys if facts[k]['state'] != 'UNREVIEWED'] +
                    [{'fact_id': 'supported_possibilities', **evidence}])
    result['applied_clauses'].append({'clause_id': 'UNCERTAINTY', 'priority': 'meta',
                                    'verbatim': table['clauses']['UNCERTAINTY']['verbatim']})
    return result


def derive_all(table, data):
    output = []
    for record in data['images'].values():
        for link in record['policy_links']:
            row = evaluate(table, record, link['policy_id'])
            row.update(review_item_id=link['review_item_id'], image_sha256=record['image_sha256'],
                       target_identifier=record['target_identifier'],
                       policy_meaning_human_validated=False,
                       source_group_verification_status=record['source_group_verification_status'])
            output.append(row)
    return output


def next_question(table, record):
    results = [evaluate(table, record, link['policy_id']) for link in record['policy_links']]
    if record.get('definition_issue'):
        return {'kind': 'DEFINITION_UNRESOLVED', 'question': None}
    needs = {k for row in results for k in row['needed_fact_ids']}
    for key in table['decision_fact_ids']:
        if key in needs:
            return {'kind': 'FACT', 'fact_id': key, **table['fact_definitions'][key]}
    if any(r['needs_supported_possibilities'] for r in results):
        if record['possibilities']['status'] == 'UNABLE_TO_SPECIFY':
            return {'kind': 'EVIDENCE_INCOMPLETE', 'question': None}
        branches = []
        for bid, branch in table['possibility_branches'].items():
            if all(record['facts'][key]['state'] not in {'TRUE', 'FALSE'} or
                   (record['facts'][key]['state'] == 'TRUE') == v for key, v in branch['when'].items()):
                branches.append({'branch_id': bid, 'description': branch['description']})
        return {'kind': 'POSSIBILITIES', 'question': '이미지 근거상 아직 남아 있는 가능성은 무엇입니까?', 'branches': branches}
    return {'kind': 'NO_ADDITIONAL_FACT_REQUIRED', 'question': None}


