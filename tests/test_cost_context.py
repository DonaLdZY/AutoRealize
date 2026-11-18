import copy
import json
from types import SimpleNamespace

from autorealize.prompt_cache import dynamic_json_block, stable_dynamic_prompt
from autorealize.review_prompt import decode_review_snapshot
from autorealize.report_writer import render_automl_context_markdown
from autorealize.models import AutoMLContextPack, ConstraintsAndTipsDraft, DescriptionSectionDraft
from autorealize.config import AutoRealizeConfig
from autorealize.modules.task_definition import TaskDefinitionModule


def decode_dynamic(text, stable):
    envelope = decode_review_snapshot(text.split('\n', 1)[1])
    if '_stable_reference_encoding' not in envelope:
        return envelope
    def resolve(value):
        if isinstance(value, dict) and set(value) == {'$stable_ref'}:
            result = stable
            for key in value['$stable_ref'].split('/')[1:]:
                key = key.replace('~1', '/').replace('~0', '~')
                result = result[int(key)] if isinstance(result, list) else result[key]
            return result
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value
    return resolve(envelope['data'])


def test_cross_message_aliases_preserve_changed_tail_and_prefix():
    rule = {'text': '容量必须不超过 500 kg；无停靠点数量上限。' * 100, 'values': [False, 0, None]}
    stable = {'a/~': [rule], 'other': rule}
    dynamic = {'again': rule, 'changed': {**rule, 'values': [False, 1, None]}, 'tail': '禁止假设缺失日期'}
    prefix, encoded = stable_dynamic_prompt(stable=stable, dynamic=dynamic)
    assert '$stable_ref' in encoded
    assert decode_dynamic(encoded, stable) == dynamic
    assert prefix == stable_dynamic_prompt(stable=stable, dynamic={'another_stage': True})[0]
    assert '禁止假设缺失日期' in encoded


def test_source_markers_and_small_inputs_stay_literal():
    value = {'$stable_ref': '/user-owned', 'text': 'x' * 1000}
    assert json.loads(dynamic_json_block('tail', value, {'v': value}).split('\n', 1)[1]) == value
    assert dynamic_json_block('tail', {'limit': 0}, {'limit': 0}) == 'tail\n{"limit":0}'


def test_prompt_projection_keeps_contracts_and_does_not_mutate_audit_pack():
    pack = AutoMLContextPack(
        execution_contract={'open_questions': ['DO_NOT_ASSUME_DATES'], 'resolved_question_history': ['OLD_HISTORY']},
        constraints=['NO_STOP_CAP'], leakage_guards=['HOLDOUT_FORBIDDEN'],
        data_access=[{'path': 'data.csv', 'fields': [{'name': 'mass (kg)', 'meaning': 'WEIGHT', 'logical_type': 'numeric',
            'numeric_stats': {'min': 0, 'max': 500, 'mean': 333}, 'top_values': ['DESCRIPTIVE_SAMPLE']}],
            'field_profiles': [{'name': 'mass (kg)', 'group_profile': 'GROUP_RANGE'}],
            'read_example': "pd.read_csv('data.csv', sep=';')", 'warnings': ['READ_WARNING']}],
    )
    before = copy.deepcopy(pack.model_dump())
    compact = render_automl_context_markdown(pack)
    audit = render_automl_context_markdown(pack, include_audit_details=True)
    assert pack.model_dump() == before
    for text in ['DO_NOT_ASSUME_DATES', 'NO_STOP_CAP', 'HOLDOUT_FORBIDDEN', 'mass (kg)', 'WEIGHT', 'GROUP_RANGE', 'READ_WARNING', "sep=';'", "'max': 500"]:
        assert text in compact
    assert 'OLD_HISTORY' not in compact and 'DESCRIPTIVE_SAMPLE' not in compact
    assert 'OLD_HISTORY' in audit and 'DESCRIPTIVE_SAMPLE' in audit


def test_compact_context_deduplicates_repeated_long_fact_but_audit_keeps_it():
    fact = "A fixed evaluation population and metric must be shared by every candidate. " * 3
    pack = AutoMLContextPack(modeling_boundary=[fact], leakage_guards=[fact])

    compact = render_automl_context_markdown(pack)
    audit = render_automl_context_markdown(pack, include_audit_details=True)

    assert compact.count("A fixed evaluation population") == 3
    assert audit.count("A fixed evaluation population") == 6


def test_constraints_and_tips_share_one_request_without_tail_clipping(tmp_path):
    calls = []
    class LLM:
        def ask_structured(self, **kwargs):
            calls.append(kwargs)
            return ConstraintsAndTipsDraft(
                constraints=DescriptionSectionDraft(markdown='- 禁止使用最终留出反馈选模'),
                tips=DescriptionSectionDraft(markdown='- 缺日期不能宣称运营达标'))
    services = SimpleNamespace(llm_client=LLM(), prompt_mgr=SimpleNamespace(load=lambda _: 'test'))
    module = TaskDefinitionModule(AutoRealizeConfig(), services, tmp_path, tmp_path)
    context = {'authoritative_memory': {'constraints': [f'RULE_{i}' for i in range(60)],
        'unresolved_questions': ['Z' * 500 + 'TAIL_QUESTION']}}
    sections = module._generate_constraints_and_tips(downstream_context=context, relations=[],
        frozen_sections={'评估协议': 'MAE; minimize'}, original_text='原始需求')
    assert len(calls) == 1
    assert 'RULE_59' in calls[0]['static_context_prompt']
    assert 'TAIL_QUESTION' in calls[0]['static_context_prompt']
    assert sections[0].section_id == 'constraints_leakage'
    assert sections[1].section_id == 'tips'
    assert sections[0].markdown.startswith('## 约束与防泄漏')
    assert sections[1].markdown.startswith('## 关键坑点与待确认事项')


def test_schema_dedup_only_references_facts_actually_visible_in_same_prompt():
    fields = [{'name':f'f{i}', 'meaning':f'EXACT_MEANING_{i}'} for i in range(18)]
    pack = AutoMLContextPack(data_schema_contract={'tables':[{'table_id':'table', 'field_summaries':fields}]},
        data_access=[{'table_id':'table', 'fields':[fields[0], fields[17], {'name':'f1','meaning':'DIFFERENT_MEANING'}]}])
    original = copy.deepcopy(pack.model_dump())
    compact = render_automl_context_markdown(pack)
    assert compact.count('EXACT_MEANING_0') == 1
    assert 'EXACT_MEANING_17' in compact  # outside the visible schema window
    assert 'EXACT_MEANING_1' in compact and 'DIFFERENT_MEANING' in compact
    assert 'only additional or differing field facts follow' in compact
    assert pack.model_dump() == original


def test_optional_legacy_profile_nulls_remain_renderable():
    for tables in (None, [{'table_id':'table', 'field_summaries':None}]):
        pack = AutoMLContextPack(data_schema_contract={'tables':tables},
            data_access=[{'table_id':'table','fields':None}])
        assert 'AutoRealize Structured Context' in render_automl_context_markdown(pack)
