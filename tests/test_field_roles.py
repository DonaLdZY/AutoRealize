import pytest

from autorealize.context_compiler import field_role


@pytest.mark.parametrize("field,expected", [
    ("订单号", "id_or_key"),
    ("原始订单号", "id_or_key"),
    ("设备编号", "id_or_key"),
    ("产品编码", "id_or_key"),
    ("实体标识", "id_or_key"),
    ("承运商代码", "id_or_key"),
    ("主键", "id_or_key"),
    ("关联外键", "id_or_key"),
    ("行序号", "id_or_key"),
    ("修改时间", "time"),
    ("订单时间", "time"),
    ("订单日期", "time"),
    ("交付时刻", "time"),
    ("销售月份", "time"),
    ("目标值", "target"),
    ("故障标签", "target"),
    ("订单金额", "cost_or_value"),
    ("合同价格", "cost_or_value"),
    ("运输费用", "cost_or_value"),
    ("合同费率", "cost_or_value"),
    ("总成本", "cost_or_value"),
    ("评价分数", "cost_or_value"),
    ("装载容量", "constraint"),
    ("订单容量", "constraint"),
    ("业务约束", "constraint"),
    ("车型限制", "constraint"),
    ("库存上限", "constraint"),
    ("温度下限", "constraint"),
    ("提交文件", "output"),
    ("订单输出", "output"),
    ("交付物", "output"),
    ("预测结果", "output"),
    ("订单", "other"),
    ("普通特征", "other"),
    ("unknown??", "other"),
])
def test_chinese_field_roles_use_specific_keywords(field, expected):
    assert field_role(field) == expected


@pytest.mark.parametrize("field,expected", [
    ("record_id", "id_or_key"),
    ("join_key", "id_or_key"),
    ("carrier_code", "id_or_key"),
    ("event_date", "time"),
    ("event_time", "time"),
    ("target", "target"),
    ("label", "target"),
    ("cost", "cost_or_value"),
    ("price", "cost_or_value"),
    ("amount", "cost_or_value"),
    ("fee", "cost_or_value"),
    ("rate", "cost_or_value"),
    ("score", "cost_or_value"),
    ("constraint", "constraint"),
    ("limit", "constraint"),
    ("capacity", "constraint"),
    ("submission", "output"),
    ("output", "output"),
    ("temperature", "other"),
])
def test_english_field_roles_keep_existing_case_insensitive_behavior(field, expected):
    assert field_role(field) == expected
    assert field_role(field.upper()) == expected


@pytest.mark.parametrize("meaning,expected", [
    ("设备的唯一标识", "id_or_key"),
    ("交付时间", "time"),
    ("故障标签", "target"),
    ("运输金额", "cost_or_value"),
    ("装载容量", "constraint"),
    ("模型输出", "output"),
])
def test_chinese_semantic_descriptions_also_inform_field_roles(meaning, expected):
    assert field_role("value", meaning) == expected
