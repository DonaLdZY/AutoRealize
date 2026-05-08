from autorealize.agents.ground_agents import GroundAgentFactory


def test_factory_creates_distinct_agent_classes() -> None:
    assert GroundAgentFactory.create("reader", ["table_io"]).__class__.__name__ == "ReaderGroundAgent"
    assert GroundAgentFactory.create("profiler", ["stats_profile"]).__class__.__name__ == "ProfilerGroundAgent"
    assert GroundAgentFactory.create("validator", ["python_sandbox", "contract_check"]).__class__.__name__ == "ValidatorGroundAgent"
    assert GroundAgentFactory.create("noop_keeper", []).__class__.__name__ == "NoopKeeperGroundAgent"
    assert GroundAgentFactory.create("repairer", ["python_sandbox"]).__class__.__name__ == "RepairerGroundAgent"
    assert GroundAgentFactory.create("transformer", ["python_sandbox"]).__class__.__name__ == "TransformerGroundAgent"
    assert GroundAgentFactory.create("join_planner", ["table_io", "stats_profile"]).__class__.__name__ == "JoinPlannerGroundAgent"
    assert GroundAgentFactory.create("schema_mapper", ["table_io", "stats_profile"]).__class__.__name__ == "SchemaMapperGroundAgent"
    assert GroundAgentFactory.create("constraint_author", ["contract_check", "constraint_engine"]).__class__.__name__ == "ConstraintAuthorGroundAgent"
    assert GroundAgentFactory.create("submission_formatter", ["table_io"]).__class__.__name__ == "SubmissionFormatterGroundAgent"
