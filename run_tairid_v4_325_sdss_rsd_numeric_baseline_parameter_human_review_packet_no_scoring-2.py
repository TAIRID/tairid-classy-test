#!/usr/bin/env python3
"""
TAIRID v4.325 SDSS RSD Numeric Baseline Parameter Human Review Packet No Scoring.

This script ingests a v4.324 parameter-source packet output directory or zip and creates a v4.325
human-review packet. It does not approve parameters, fill numeric baselines, load observed values,
or score observed data.
"""
import argparse, json, zipfile, shutil, tempfile
from pathlib import Path
import pandas as pd

TRUTH_BOUNDARY = "no observed scoring; no parameter approval; no numeric baseline approval; no support claims"


def find_input_dir(path: Path) -> Path:
    if path.is_dir():
        candidates = list(path.rglob('v4_324_summary.json'))
        if not candidates:
            raise FileNotFoundError('Could not find v4_324_summary.json in input directory')
        return candidates[0].parent
    if path.suffix.lower() == '.zip':
        tmp = Path(tempfile.mkdtemp(prefix='tairid_v4_325_input_'))
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(tmp)
        nested = list(tmp.rglob('v4_324_summary.json'))
        if not nested:
            nested_zips = list(tmp.rglob('*.zip'))
            for nz in nested_zips:
                try:
                    with zipfile.ZipFile(nz, 'r') as z2:
                        z2.extractall(tmp / (nz.stem + '_unzipped'))
                except zipfile.BadZipFile:
                    continue
            nested = list(tmp.rglob('v4_324_summary.json'))
        if not nested:
            raise FileNotFoundError('Could not find v4_324_summary.json inside input zip')
        return nested[0].parent
    raise ValueError('Input must be a directory or zip file')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='tairid-v4-324-sdss-rsd-numeric-baseline-parameter-source-packet-output.zip')
    ap.add_argument('--output-dir', default='tairid_v4_325_sdss_rsd_numeric_baseline_parameter_human_review_packet_no_scoring_output')
    args = ap.parse_args()
    input_path = Path(args.input)
    base = find_input_dir(input_path)
    outdir = Path(args.output_dir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    human_324 = pd.read_csv(base/'SDSS_RSD_NUMERIC_BASELINE_PARAMETER_HUMAN_REVIEW_PACKET_v4_324_NO_APPROVAL.csv')
    source_324 = pd.read_csv(base/'SDSS_RSD_NUMERIC_BASELINE_PARAMETER_SOURCE_PACKET_v4_324_NO_APPROVAL.csv')
    summary_324 = json.loads((base/'v4_324_summary.json').read_text())

    triage_rows=[]
    for i,row in human_324.reset_index(drop=True).iterrows():
        triage_rows.append({
            'review_id_v4_325': f'parameter_human_review_325_{i+1:03d}',
            'inherited_review_id_v4_324': row['review_id'],
            'question_id': row['question_id'],
            'parameter_or_method': row['parameter_or_method'],
            'question': row['question'],
            'candidate_id': row['candidate_id'],
            'candidate_class': row['candidate_class'],
            'source_url': row['source_url'],
            'candidate_quote_available': bool(row['candidate_quote_available']),
            'candidate_quote_window': row.get('candidate_quote_window',''),
            'candidate_status_entering_v4_325': 'not_approved',
            'human_triage_decision': '',
            'allowed_triage_decisions': 'advance_to_later_parameter_value_review|needs_better_source|reject_candidate|duplicate_context_only|keep_blocked',
            'human_selected_basis_text': '',
            'human_reviewer_note': '',
            'v4_325_action_scope': 'source_candidate_triage_only_no_values',
            'parameter_value_approval_allowed_now': False,
            'numeric_baseline_approval_allowed_now': False,
            'observed_values_loaded': False,
            'definition_match_confirmed': False,
            'score_allowed': False,
            'support_allowed': False,
            'claim_limit_must_remain': 'no_observed_scoring/no_parameter_approval/no_numeric_baseline_approval/no_support_claims',
        })
    triage = pd.DataFrame(triage_rows)

    decision_rows=[]
    for i,(qid, grp) in enumerate(triage.groupby('question_id', sort=True), start=1):
        decision_rows.append({
            'decision_id_v4_325': f'parameter_candidate_decision_325_{i:03d}',
            'question_id': qid,
            'parameter_or_method': grp['parameter_or_method'].iloc[0],
            'review_question': grp['question'].iloc[0],
            'human_decision': '',
            'allowed_decisions': 'select_candidate_for_later_value_review|needs_better_source|reject_all|keep_blocked',
            'allowed_candidate_ids_from_packet': '|'.join(grp['candidate_id'].tolist()),
            'human_selected_candidate_ids': '',
            'source_urls_available_for_review': '|'.join(sorted(set(grp['source_url'].astype(str).tolist()))),
            'quoted_or_precise_source_text_for_selected_candidate': '',
            'human_answer_summary_no_values': '',
            'reviewer_note': '',
            'decision_scope': 'candidate_source_triage_only_no_parameter_values',
            'claim_limit_must_remain': 'no_scoring/no_claims/no_parameter_value_approval/no_numeric_baseline_approval',
            'parameter_value_approval_allowed_now': False,
            'numeric_baseline_approval_allowed_now': False,
            'observed_values_loaded': False,
            'definition_match_confirmed': False,
            'score_allowed': False,
            'support_allowed': False,
        })
    decision_template = pd.DataFrame(decision_rows)

    question_summary = triage.groupby(['question_id','parameter_or_method'], sort=True).agg(
        candidate_rows_in_review=('candidate_id','count'),
        candidate_quote_available_rows=('candidate_quote_available','sum'),
        candidate_ids=('candidate_id', lambda s: '|'.join(s)),
        source_urls=('source_url', lambda s: '|'.join(sorted(set(map(str,s)))))
    ).reset_index()
    for col in ['human_decision_filled_now','parameter_value_approved_now','numeric_baseline_approved_now','score_allowed','support_allowed']:
        question_summary[col] = False

    review_ids=set(triage['candidate_id'].tolist())
    ledger = source_324.copy()
    ledger.insert(0,'ledger_id_v4_325',[f'parameter_source_ledger_325_{i+1:03d}' for i in range(len(ledger))])
    ledger['included_in_v4_325_human_review_packet'] = ledger['candidate_id'].isin(review_ids)
    ledger['v4_325_review_scope'] = ledger['included_in_v4_325_human_review_packet'].map({True:'human_candidate_triage_only_no_values', False:'retained_as_unapproved_source_candidate_not_in_current_triage'})
    for col in ['parameter_value_approved_now','numeric_baseline_approved_now','observed_values_loaded','definition_match_confirmed','score_allowed','support_allowed']:
        ledger[col] = False
    ledger['claim_limit_must_remain'] = 'no_scoring/no_claims/no_parameter_value_approval/no_numeric_baseline_approval'

    checklist = pd.DataFrame([
        {'check_id':'v4_325_check_001','human_review_check':'Does the candidate source actually justify moving into a later value-review step, rather than merely mentioning related cosmology words?','required':True},
        {'check_id':'v4_325_check_002','human_review_check':'Does the decision avoid selecting or approving any numeric value for Omega_m0, sigma8_0, gamma, or any baseline calculation?','required':True},
        {'check_id':'v4_325_check_003','human_review_check':'Does the decision keep observed SDSS RSD scoring blocked?','required':True},
        {'check_id':'v4_325_check_004','human_review_check':'Does the decision keep definition-match, covariance, residual, chi-square, p-value, likelihood, and model-support claims blocked?','required':True},
        {'check_id':'v4_325_check_005','human_review_check':'Does the decision preserve no_scoring/no_claims language in every output file?','required':True},
        {'check_id':'v4_325_check_006','human_review_check':'Does the decision separate source-candidate triage from later parameter-value approval?','required':True},
        {'check_id':'v4_325_check_007','human_review_check':'Does the decision avoid treating dry-run calculator context as an approved numeric baseline?','required':True},
        {'check_id':'v4_325_check_008','human_review_check':'Does the decision preserve all v4.324 false gates before any future step can run?','required':True},
    ])

    rules=[]
    def add(gate, passed): rules.append({'gate':gate,'passed':bool(passed)})
    add('input_v4_324_summary_found', summary_324.get('version')=='v4.324')
    add('input_v4_324_rules_passed_8_of_8', summary_324.get('rules_passed')==8 and summary_324.get('rules_total')==8)
    add('input_v4_324_no_parameter_approval', summary_324.get('parameter_values_approved') is False)
    add('input_v4_324_no_numeric_baseline_approval', summary_324.get('numeric_baseline_values_filled') is False and summary_324.get('approved_baseline_values')==0)
    add('input_v4_324_no_observed_scoring', summary_324.get('observed_values_loaded') is False and summary_324.get('score_valid') is False)
    add('input_v4_324_no_support_allowed', summary_324.get('support_allowed') is False)
    add('source_candidate_rows_equal_48', len(source_324)==48)
    add('candidate_review_rows_equal_24', len(human_324)==24)
    add('v4_325_triage_rows_equal_24', len(triage)==24)
    add('v4_325_decision_template_rows_equal_6', len(decision_template)==6)
    add('v4_325_no_human_decisions_prefilled', (triage['human_triage_decision'].fillna('')=='').all() and (decision_template['human_decision'].fillna('')=='').all())
    add('v4_325_no_parameter_value_approval', not triage['parameter_value_approval_allowed_now'].any() and not decision_template['parameter_value_approval_allowed_now'].any() and not ledger['parameter_value_approved_now'].any())
    add('v4_325_no_numeric_baseline_approval', not triage['numeric_baseline_approval_allowed_now'].any() and not decision_template['numeric_baseline_approval_allowed_now'].any() and not ledger['numeric_baseline_approved_now'].any())
    add('v4_325_no_observed_values_loaded', not triage['observed_values_loaded'].any() and not decision_template['observed_values_loaded'].any() and not ledger['observed_values_loaded'].any())
    add('v4_325_no_definition_match_confirmed', not triage['definition_match_confirmed'].any() and not decision_template['definition_match_confirmed'].any() and not ledger['definition_match_confirmed'].any())
    add('v4_325_score_blocked', not triage['score_allowed'].any() and not decision_template['score_allowed'].any() and not ledger['score_allowed'].any())
    add('v4_325_support_blocked', not triage['support_allowed'].any() and not decision_template['support_allowed'].any() and not ledger['support_allowed'].any())
    rule_checks = pd.DataFrame(rules)
    failed = rule_checks.loc[~rule_checks['passed'],'gate'].tolist()
    summary = {
        'version':'v4.325',
        'route_name':'v4_325_sdss_rsd_numeric_baseline_parameter_human_review_packet_no_scoring',
        'expected_result':'v4_325_sdss_rsd_numeric_baseline_parameter_human_review_packet_no_scoring_passed_no_claims',
        'input_version':'v4.324',
        'rules_passed':int(rule_checks['passed'].sum()),
        'rules_total':int(len(rule_checks)),
        'failed_gates':'none' if not failed else '|'.join(failed),
        'input_source_candidate_rows':int(len(source_324)),
        'input_candidate_review_rows':int(len(human_324)),
        'candidate_quote_available_rows_in_v4_325_review':int(triage['candidate_quote_available'].astype(bool).sum()),
        'v4_325_triage_rows':int(len(triage)),
        'v4_325_decision_template_rows':int(len(decision_template)),
        'v4_325_question_summary_rows':int(len(question_summary)),
        'parameter_values_approved':False,
        'numeric_baseline_values_filled':False,
        'approved_baseline_values':0,
        'observed_values_loaded':False,
        'definition_match_confirmed':False,
        'score_valid':False,
        'support_allowed':False,
        'claim_boundary':TRUTH_BOUNDARY,
        'selected_next_route':'v4_326_sdss_rsd_numeric_baseline_parameter_decision_ingest_no_scoring_if_human_template_filled',
    }

    triage.to_csv(outdir/'SDSS_RSD_NUMERIC_BASELINE_PARAMETER_HUMAN_REVIEW_PACKET_v4_325_NO_APPROVAL_NO_SCORING.csv', index=False)
    decision_template.to_csv(outdir/'SDSS_RSD_NUMERIC_BASELINE_PARAMETER_CANDIDATE_DECISION_TEMPLATE_v4_325_FILLABLE_NO_VALUES.csv', index=False)
    question_summary.to_csv(outdir/'v4_325_question_review_summary_no_approval.csv', index=False)
    ledger.to_csv(outdir/'v4_325_parameter_source_candidate_ledger_no_approval.csv', index=False)
    checklist.to_csv(outdir/'v4_325_human_review_checklist_rows.csv', index=False)
    rule_checks.to_csv(outdir/'v4_325_rule_checks.csv', index=False)
    (outdir/'v4_325_summary.json').write_text(json.dumps(summary, indent=2))
    (outdir/'v4_325_summary.md').write_text(f"""# TAIRID v4.325 SDSS RSD Numeric Baseline Parameter Human Review Packet No Scoring

Expected result: `{summary['expected_result']}`

{summary['rules_passed']} / {summary['rules_total']} rules passed

failed_gates = {summary['failed_gates']}

Truth boundary: {TRUTH_BOUNDARY}.

The following remain false:

```text
parameter_values_approved = False
numeric_baseline_values_filled = False
approved_baseline_values = 0
observed_values_loaded = False
definition_match_confirmed = False
score_valid = False
support_allowed = False
```

Next clean route: `{summary['selected_next_route']}`
""")

    zip_path = outdir.with_suffix('.zip')
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(outdir.rglob('*')):
            z.write(p, p.relative_to(outdir.parent))
    print(json.dumps(summary, indent=2))
    print(f'WROTE {zip_path}')

if __name__ == '__main__':
    main()
