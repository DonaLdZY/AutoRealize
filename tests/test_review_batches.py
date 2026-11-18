import json
import pytest

from autorealize.review_prompt import bounded_review_batches, decode_review_snapshot


def _decode_block(block):
    return decode_review_snapshot(block.split("\n", 1)[1])


def _merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        elif key not in target:
            target[key] = value
        else:
            assert target[key] == value


def test_batches_preserve_every_leaf_and_repeat_only_bounded_shared_context():
    authority = {'original_requirements_full':'AUTHORITATIVE', 'evidence':{'first':'A'*1500,'last':'B'*1500}}
    stage_evidence = {'stable_facts': {'one': 'C' * 1500, 'two': 'D' * 1500}}
    request = {'description_markdown':'CURRENT DESCRIPTION','evaluation_contract':{'metric':'MAE'},
        'description_protocol_bundle':{'horizon':'1 hour'}, 'artifact':{str(i):f'UNIQUE_{i}_' + str(i)*1500 for i in range(7)},'round':1}
    batches = bounded_review_batches(authority=authority,stage_evidence=stage_evidence,latest_request=request,dynamic_title='audit',input_budget=1800)
    assert len(batches)>1
    all_paths=[]
    shard_paths=[]
    shared_paths=set()
    recovered_authority={}
    recovered_stage_evidence={}
    recovered_request={}
    recovered_evaluation=False
    recovered_protocol=False
    for batch in batches:
        assert batch['input_tokens'] <= 1800
        snapshot=_decode_block(batch['stable'])
        assert snapshot['authority']['original_requirements_full']=='AUTHORITATIVE'
        assert snapshot['latest_request']['description_markdown']=='CURRENT DESCRIPTION'
        dynamic=_decode_block(batch['dynamic'])
        batch_evidence=dynamic['batch_evidence']
        scope=batch_evidence['review_batch_scope']
        assert 'authority/original_requirements_full' in scope['shared_paths']
        shared_paths.update(scope['shared_paths'])
        evidence=batch_evidence['evidence']
        _merge(recovered_authority,evidence['authority'])
        _merge(recovered_stage_evidence,evidence['stage_evidence'])
        _merge(recovered_request,evidence['latest_request'])
        recovered_evaluation |= evidence['latest_request'].get('evaluation_contract',{}).get('metric')=='MAE'
        recovered_protocol |= evidence['latest_request'].get('description_protocol_bundle',{}).get('horizon')=='1 hour'
        all_paths.extend(batch['included_paths'])
        shard_paths.extend(
            path for path in batch['included_paths']
            if path not in scope['shared_paths']
        )
    assert len({batch['stable'] for batch in batches}) == 1
    assert recovered_authority['evidence']==authority['evidence']
    assert recovered_stage_evidence == stage_evidence
    assert recovered_request['artifact']==request['artifact']
    assert recovered_evaluation and recovered_protocol
    assert any(p.startswith('authority/evidence') for p in all_paths)
    assert len(shard_paths)==len(set(shard_paths))
    assert all_paths.count('authority/original_requirements_full')==len(batches)
    assert shared_paths <= set(all_paths)


def test_dynamic_batch_evidence_does_not_change_lossless_shared_prefix():
    authority={'original_requirements_full':'authoritative request'}
    request={'description_markdown':'current complete task', 'round':1}
    first=bounded_review_batches(authority=authority,stage_evidence={'rules':'fixed'},
        latest_request={**request,'one':{'evidence':'A'*500}},dynamic_title='audit',input_budget=4000)[0]
    second=bounded_review_batches(authority=authority,stage_evidence={'rules':'fixed'},
        latest_request={**request,'one':{'evidence':'B'*500}},dynamic_title='audit',input_budget=4000)[0]
    assert first['stable']==second['stable']
    first_dynamic=_decode_block(first['dynamic'])
    second_dynamic=_decode_block(second['dynamic'])
    assert first_dynamic['batch_evidence']['evidence']['latest_request']['one']['evidence']=='A'*500
    assert second_dynamic['batch_evidence']['evidence']['latest_request']['one']['evidence']=='B'*500


def test_large_stage_evidence_is_partitioned_instead_of_repeated():
    stage_evidence = {
        "artifact_audit": {str(i): f"evidence-{i}-" + "x" * 9000 for i in range(8)}
    }
    batches = bounded_review_batches(
        authority={"original_requirements_full": "authoritative task"},
        stage_evidence=stage_evidence,
        latest_request={"description_markdown": "current task"},
        dynamic_title="audit",
        input_budget=5000,
    )
    assert len(batches) > 1
    recovered = {}
    for batch in batches:
        assert batch["input_tokens"] <= 5000
        evidence = _decode_block(batch["dynamic"])["batch_evidence"]["evidence"]
        _merge(recovered, evidence["stage_evidence"])
        assert _decode_block(batch["stable"])["stage_evidence"] == {}
    assert recovered == stage_evidence


def test_48k_audit_fits_one_batch_with_64k_context_budget():
    request = {
        "description_markdown": "CURRENT DESCRIPTION",
        "complete_artifact_snapshot": "evidence " * 19000,
        "round": 1,
    }

    batches = bounded_review_batches(
        authority={"original_requirements_full": "authoritative task"},
        stage_evidence={},
        latest_request=request,
        dynamic_title="audit",
        input_budget=64000,
    )

    assert len(batches) == 1
    assert batches[0]["input_tokens"] < 64000
    assert "latest_request/complete_artifact_snapshot" in batches[0]["included_paths"]


def test_final_review_sends_description_body_once_and_keeps_section_index(tmp_path):
    from types import SimpleNamespace
    from autorealize.config import AutoRealizeConfig
    from autorealize.models import (
        ArtifactConsistencyReview,
        DescriptionProtocolBundle,
        EvaluationContractReview,
        ProblemParadigmReview,
        SampleSubmissionSpec,
    )
    from autorealize.modules.task_definition import TaskDefinitionModule

    requests=[]

    class LLM:
        def ask_structured(self, **kwargs):
            requests.append(kwargs)
            return ArtifactConsistencyReview(passed=True, summary='consistent', issues=[])

    config=AutoRealizeConfig()
    config.prompt.lossless_review_prompts=True
    module=TaskDefinitionModule(config,SimpleNamespace(llm_client=LLM(),
        prompt_mgr=SimpleNamespace(load=lambda _:'audit')),tmp_path,tmp_path)
    body='UNIQUE_DESCRIPTION_BODY ' * 120
    module._review_final_artifact_consistency(
        desc=body, problem_review=ProblemParadigmReview(), protocol_bundle=DescriptionProtocolBundle(),
        evaluation_contract=EvaluationContractReview(), sample_spec=SampleSubmissionSpec(),
        submission_report={}, automl_context_pack={}, main_task_protocol={
            'frozen_description_sections': {'data': {'section_id':'data','markdown':body}}},
        downstream_context={}, deterministic_defects=[], round_idx=1)

    stable=_decode_block(requests[0]['static_context_prompt'])
    assert stable['latest_request']['description_markdown']==body
    protocol=stable['latest_request']['main_task_protocol']
    assert protocol['frozen_description_sections']=={
        'data': {'section_id':'data','description_ref':'description_markdown::data'}}
    assert body not in json.dumps(protocol,ensure_ascii=False)


def test_indivisible_evidence_is_never_truncated():
    with pytest.raises(ValueError,match='Indivisible'):
        bounded_review_batches(authority={},stage_evidence={},latest_request={'huge':'x'*10000},dynamic_title='audit',input_budget=1000)


def test_large_lists_are_partitioned_without_losing_leaf_values():
    authority = {'original_requirements_full': 'AUTHORITATIVE'}
    request = {
        'description_markdown': 'CURRENT DESCRIPTION',
        'data_schema_contract': {
            'tables': [
                {'name': f'table_{i}', 'columns': [f'column_{i}_{j}' for j in range(8)]}
                for i in range(40)
            ]
        },
        'round': 1,
    }
    batches = bounded_review_batches(
        authority=authority, stage_evidence={}, latest_request=request,
        dynamic_title='audit', input_budget=1200,
    )
    assert len(batches) > 1
    recovered = {'tables': []}
    shard_paths = []
    for batch in batches:
        assert batch['input_tokens'] <= 1200
        dynamic = _decode_block(batch['dynamic'])
        batch_evidence = dynamic['batch_evidence']
        evidence = batch_evidence['evidence']['latest_request']
        tables = evidence.get('data_schema_contract', {}).get('tables', [])
        recovered['tables'].extend(item for item in tables if item is not None)
        shard_paths.extend(
            path for path in batch['included_paths']
            if path not in batch_evidence['review_batch_scope']['shared_paths']
        )
    assert recovered['tables'] == request['data_schema_contract']['tables']
    assert len(shard_paths) == len(set(shard_paths))


def test_shared_contract_that_does_not_fit_fails_explicitly():
    with pytest.raises(ValueError,match='Common authoritative'):
        bounded_review_batches(authority={'original_requirements_full':'x'*10000},stage_evidence={},latest_request={},dynamic_title='audit',input_budget=1000)


def test_full_scan_evidence_is_shared_across_every_review_batch():
    scan_evidence = [{
        "source_file": "device.csv",
        "source_fingerprint": "sha256:device",
        "full_table_scan": {
            "status": "completed",
            "rows_scanned": 10,
            "valid_timestamp_rows": 10,
            "invalid_timestamp_rows": 0,
        },
    }]
    batches = bounded_review_batches(
        authority={"original_requirements_full": "Scan the complete device table."},
        stage_evidence={
            "full_scan_evidence": scan_evidence,
            "large_audit_facts": {
                str(index): f"FACT_{index}_" + ("x" * 2000)
                for index in range(8)
            },
        },
        latest_request={"description_markdown": "The full scan verified ten records.", "round": 1},
        dynamic_title="audit",
        input_budget=3000,
    )

    assert len(batches) > 1
    for batch in batches:
        dynamic = _decode_block(batch["dynamic"])
        evidence = dynamic["batch_evidence"]["evidence"]
        assert evidence["stage_evidence"]["full_scan_evidence"] == scan_evidence
        assert "stage_evidence/full_scan_evidence" in dynamic["batch_evidence"]["review_batch_scope"]["included_paths"]
        assert batch["input_tokens"] <= 3000


def test_one_failed_batch_prevents_total_approval(tmp_path):
    from types import SimpleNamespace
    from pathlib import Path
    from autorealize.config import AutoRealizeConfig
    from autorealize.models import ArtifactConsistencyReview, ArtifactConsistencyIssue, DescriptionProtocolBundle, EvaluationContractReview, ProblemParadigmReview, SampleSubmissionSpec
    from autorealize.modules.task_definition import TaskDefinitionModule
    calls=[]
    class LLM:
        def ask_structured(self,**kwargs):
            calls.append(kwargs)
            failed=len(calls)==2
            return ArtifactConsistencyReview(passed=not failed,summary='fixture review',issues=[ArtifactConsistencyIssue(
                issue_id='different_population',severity='blocking',artifact='automl_context_pack',section='evaluation_contract',
                repair_target='machine_contract',message='development population differs from frozen contract')] if failed else [])
    config=AutoRealizeConfig()
    config.llm.context_window_tokens=1_000_000
    prompt_root = Path(__file__).parents[1] / "autorealize" / "prompts"
    def load_prompt(name):
        if name == "system/artifact_consistency_batch_reviewer.md":
            return (prompt_root / name).read_text(encoding="utf-8")
        return "audit"
    module=TaskDefinitionModule(config,SimpleNamespace(llm_client=LLM(),prompt_mgr=SimpleNamespace(load=load_prompt)),tmp_path,tmp_path)
    tables={str(i):{"physical_columns_exact":["UNIQUE_"+str(i)+"_" + str(i)*20000]} for i in range(20)}
    review=module._review_final_artifact_consistency(desc='Current task',problem_review=ProblemParadigmReview(),
        protocol_bundle=DescriptionProtocolBundle(),evaluation_contract=EvaluationContractReview(),sample_spec=SampleSubmissionSpec(),
        submission_report={},automl_context_pack={"data_schema_contract":{"tables":tables}},main_task_protocol={},
        downstream_context={},deterministic_defects=[],round_idx=1)
    assert len(calls)>2
    assert not review.passed
    assert len(review.issues)==1 and review.issues[0].issue_id.startswith('b2_')
    assert all(call["max_tokens"] == 12288 for call in calls)
    assert all(call["length_retry_max_tokens"] == 16384 for call in calls)
    receipt=json.loads((tmp_path/'artifact_consistency_batches_round_1.json').read_text())
    assert receipt['batch_count']==receipt['completed_batches']==len(calls)
    assert receipt['input_budget']==64000
    assert all('完整审查的一份无损证据分片' in call['system_prompt'] for call in calls)
