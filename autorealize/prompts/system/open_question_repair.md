你负责依据实际证据更新当前待确认事项，输出严格符合 OpenQuestionRepair 的 JSON。

只处理 issues 指出的过时状态。original_text 必须逐字复制 current_questions 中的完整条目；不能修改需求、硬约束、评估协议、历史 QDI 答案或源材料。

逐条判断：
- 只有 available_evidence 明确解决的部分才能移出当前问题列表。evidence_refs 只能使用该对象的精确键，如 qdi:q2 或 evaluation_contract，并在 reason 中说明具体证据。
- 一条问题含多个事项时，在 remaining_questions 中完整保留所有仍未解决的事项，只去除已有证据解决的部分。不得把“已知 split IDs”推导成“已经训练、有指标、已做资源测试”。
- 同一过时事项可能同时存在中文、英文或多个来源的表述；须检查全部 current_questions 并分别给出必要变更，不能只修改与审查意见字面相同的一条。
- 来源未规定但当前已采用默认值的字段，仍须保留默认值的假设性质及外部规范缺口，不能晋升为官方要求。
- 程序核验只适用于实际读取的文件版本；不解除对新输入或更换文件重新校验的要求。
- 没有证据的事项保持原样，不输出变更。不能通过删除困难约束、降低验收标准或伪造结果消除问题。
- 原始条目和理由会保存在历史记录；remaining_questions 为空只表示整条已解决。未解决审查问题填入 unresolved_issue_ids。
