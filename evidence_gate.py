from .rule_mapper import evaluate


def resolve(table, record, policy_id):
    raw = evaluate(table, record, policy_id)
    label = raw['rule_derived_label']
    status = 'ADDITIONAL_OBSERVATION_REQUIRED'
    reason = list(raw['unresolved_issues'])
    if raw['status'] == 'POLICY_DEFINITION_UNRESOLVED':
        status = 'POLICY_DEFINITION_UNRESOLVED'
    elif raw['status'] == 'HUMAN_FACT_CONFLICT':
        status = 'HUMAN_FACT_CONFLICT'
    elif raw['inference_basis'] == 'HUMAN_SUPPORTED_POSSIBILITIES_UNDER_ORIGINAL_UNCERTAINTY_RULE':
        status = 'EVIDENCE_SUPPORT_UNRESOLVED'
        reason = ['A selected possibility subset and nonempty rationale alone do not verify why other possibilities were excluded.']
    elif raw['status'] == 'RULE_DERIVED_COMPLETE' and not record.get('definition_issue'):
        status = 'RESOLVED_CONDITIONAL_ON_RECORDED_HUMAN_FACTS'
    elif raw['needs_supported_possibilities']:
        status = 'EVIDENCE_SUPPORT_UNRESOLVED'
    return {
        **raw, 'raw_rule_output': label, 'raw_rule_status': raw['status'],
        'rule_resolution_status': status,
        'y_rule_derived_for_evaluation': label if status == 'RESOLVED_CONDITIONAL_ON_RECORDED_HUMAN_FACTS' else None,
        'evaluation_exclusion_reasons': reason,
        'evidence_gate_version': 'BATCH02_EVIDENCE_GATE_v1',
        'not_independent_human_gt': True,
    }


def derive_all_gated(table, data):
    output = []
    for record in data['images'].values():
        for link in record['policy_links']:
            result = resolve(table, record, link['policy_id'])
            result.update(review_item_id=link['review_item_id'], image_sha256=record['image_sha256'],
                          target_identifier=record['target_identifier'], policy_sha256=link['policy_sha256'])
            output.append(result)
    return output


