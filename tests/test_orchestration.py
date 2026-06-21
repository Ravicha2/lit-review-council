"""
TDD tests for multi-topic orchestration (issue #7).
These tests define behavior through public interfaces.
They will fail until the implementation exists.
"""
import sys
import json
import pytest
import yaml
import tempfile
import os
from pathlib import Path

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)


# ---------------------------------------------------------------------------
# 1. TopicConfig: YAML topic config parsing and validation
# ---------------------------------------------------------------------------

class TestTopicConfig:
    def test_parse_valid_yaml_with_all_fields(self):
        from src.orchestration import TopicConfig
        config_yaml = """
topics:
  - slug: truth-maintenance
    description: Truth maintenance systems for incremental belief revision
    search_keywords:
      - truth maintenance system
      - JTMS
      - ATMS
    rationale: Foundational for understanding constraint verdicts. Groups C and F depend on this.
"""
        data = yaml.safe_load(config_yaml)
        topics = [TopicConfig(**t) for t in data["topics"]]
        assert len(topics) == 1
        assert topics[0].slug == "truth-maintenance"
        assert topics[0].rationale is not None

    def test_parse_valid_yaml_without_rationale(self):
        from src.orchestration import TopicConfig
        config_yaml = """
topics:
  - slug: incremental-recomputation
    description: Incremental graph recomputation methods
    search_keywords:
      - incremental Datalog evaluation
      - delta processing
"""
        data = yaml.safe_load(config_yaml)
        topics = [TopicConfig(**t) for t in data["topics"]]
        assert len(topics) == 1
        assert topics[0].rationale is None

    def test_slug_required(self):
        from src.orchestration import TopicConfig
        with pytest.raises(Exception):
            TopicConfig(description="desc", search_keywords=["k1"])

    def test_description_required(self):
        from src.orchestration import TopicConfig
        with pytest.raises(Exception):
            TopicConfig(slug="s", search_keywords=["k1"])

    def test_search_keywords_required(self):
        from src.orchestration import TopicConfig
        with pytest.raises(Exception):
            TopicConfig(slug="s", description="desc")

    def test_load_config_from_file(self):
        from src.orchestration import load_config
        config_content = """
topics:
  - slug: topic-a
    description: Topic A
    search_keywords:
      - keyword a1
  - slug: topic-b
    description: Topic B
    search_keywords:
      - keyword b1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()
            try:
                config = load_config(f.name)
                assert len(config.topics) == 2
                assert config.topics[0].slug == "topic-a"
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# 2. PipelinePlan: wave assignment and dependency mapping
# ---------------------------------------------------------------------------

class TestPipelinePlan:
    def test_valid_plan_with_both_waves(self):
        from src.orchestration import PipelinePlan
        plan = PipelinePlan(
            wave1=["a", "b", "d", "e"],
            wave2={"c": ["a"], "f": ["a"], "g": ["a", "c"]}
        )
        assert "a" in plan.wave1
        assert plan.wave2["c"] == ["a"]
        assert plan.wave2["g"] == ["a", "c"]

    def test_plan_with_empty_wave2(self):
        from src.orchestration import PipelinePlan
        plan = PipelinePlan(wave1=["a", "b"], wave2={})
        assert len(plan.wave1) == 2
        assert len(plan.wave2) == 0

    def test_wave2_dependencies_must_reference_wave1(self):
        from src.orchestration import PipelinePlan
        with pytest.raises(Exception):
            PipelinePlan(
                wave1=["a", "b"],
                wave2={"c": ["z"]}  # "z" not in wave1
            )

    def test_wave2_max_3_dependencies(self):
        from src.orchestration import PipelinePlan
        with pytest.raises(Exception):
            PipelinePlan(
                wave1=["a", "b", "d", "e"],
                wave2={"c": ["a", "b", "d", "e"]}  # 4 deps exceeds cap
            )

    def test_all_topic_slugs_covered(self):
        from src.orchestration import PipelinePlan
        plan = PipelinePlan(
            wave1=["a", "b", "d", "e"],
            wave2={"c": ["a"], "f": ["a"], "g": ["a", "c"]}
        )
        all_topics = set(plan.wave1) | set(plan.wave2.keys())
        assert all_topics == {"a", "b", "c", "d", "e", "f", "g"}

    def test_no_topic_in_both_waves(self):
        from src.orchestration import PipelinePlan
        with pytest.raises(Exception):
            PipelinePlan(
                wave1=["a", "b", "c"],
                wave2={"c": ["a"]}  # "c" in both waves
            )


# ---------------------------------------------------------------------------
# 3. DistilledContext: schema for context handoff between waves
# ---------------------------------------------------------------------------

class TestDistilledContext:
    def test_valid_distillation(self):
        from src.orchestration import DistilledContext
        ctx = DistilledContext(
            key_terms=["JTMS", "belief revision", "non-monotonic reasoning"],
            conclusion="Truth maintenance systems provide a mechanism for tracking which conclusions remain valid as premises change.",
            top_urls=[
                {"title": "JTMS Paper", "url": "https://arxiv.org/123", "source_tier": "peer_reviewed"}
            ]
        )
        assert len(ctx.key_terms) == 3
        assert "JTMS" in ctx.key_terms

    def test_top_urls_capped_at_5(self):
        from src.orchestration import DistilledContext
        urls = [
            {"title": f"Source {i}", "url": f"https://example.com/{i}", "source_tier": "blog_or_forum"}
            for i in range(10)
        ]
        with pytest.raises(Exception):
            DistilledContext(
                key_terms=["term"],
                conclusion="short conclusion",
                top_urls=urls  # 10 URLs exceeds cap of 5
            )

    def test_conclusion_is_required(self):
        from src.orchestration import DistilledContext
        with pytest.raises(Exception):
            DistilledContext(
                key_terms=["term"],
                top_urls=[]
            )


# ---------------------------------------------------------------------------
# 4. Orchestrator: wave execution and failure handling
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_wave1_topics_run_in_parallel(self):
        """Wave 1 topics should all start before any completes (conceptual test)."""
        from src.orchestration import build_execution_plan, PipelinePlan
        plan = PipelinePlan(
            wave1=["a", "b", "d", "e"],
            wave2={"c": ["a"], "f": ["a"]}
        )
        exec_plan = build_execution_plan(plan)
        assert len(exec_plan.wave1_topics) == 4

    def test_wave2_topics_receive_correct_prior_context(self):
        """Each Wave 2 topic gets distillations from its Wave 1 dependencies only."""
        from src.orchestration import build_execution_plan, PipelinePlan
        plan = PipelinePlan(
            wave1=["a", "b", "d", "e"],
            wave2={"c": ["a"], "f": ["a"], "g": ["a", "c"]}
        )
        # ponytail: c depends on a only, g depends on a + c (but c is wave2, only a is wave1 dep)
        exec_plan = build_execution_plan(plan)
        assert exec_plan.wave2_contexts["c"] == ["a"]
        assert exec_plan.wave2_contexts["f"] == ["a"]
        # g depends on a and c, but c is wave2 - only a's distillation is available
        assert exec_plan.wave2_contexts["g"] == ["a"]

    def test_wave1_failure_produces_empty_distillation(self):
        """If a Wave 1 topic fails, its distillation is empty string, not an error."""
        from src.orchestration import handle_wave1_failure
        result = handle_wave1_failure(topic_slug="a", error=ValueError("synthesis failed"))
        assert result.key_terms == []
        assert result.conclusion == ""
        assert result.top_urls == []

    def test_wave2_runs_without_failed_dependency_context(self):
        """Wave 2 topics still run even if a dependency failed."""
        from src.orchestration import build_prior_context_string, DistilledContext
        distillations = {
            "a": DistilledContext(
                key_terms=["JTMS"], conclusion="Some conclusion",
                top_urls=[{"title": "P", "url": "https://arxiv.org/1", "source_tier": "peer_reviewed"}]
            ),
            "b": None  # b failed
        }
        # When b failed, its entry is None. The context builder should skip it.
        context = build_prior_context_string(["a", "b"], distillations)
        assert "JTMS" in context
        assert "Some conclusion" in context

    def test_prior_context_string_respects_token_cap(self):
        """Max 3 distillations per Wave 2 topic."""
        from src.orchestration import build_prior_context_string, DistilledContext
        distillations = {}
        for slug in ["a", "b", "d", "e"]:
            distillations[slug] = DistilledContext(
                key_terms=[f"term_{slug}"], conclusion=f"Conclusion {slug}",
                top_urls=[]
            )
        # Requesting 4 distillations should only include the first 3
        context = build_prior_context_string(["a", "b", "d", "e"], distillations)
        assert "term_a" in context
        assert "term_b" in context
        assert "term_d" in context
        assert "term_e" not in context  # 4th distillation excluded by cap


# ---------------------------------------------------------------------------
# 5. Explorer prompt: prior context injection
# ---------------------------------------------------------------------------

class TestExplorerPrompt:
    def test_no_prior_context_when_none(self):
        from src.prompts import EXPLORER_INSTRUCTION_TEMPLATE
        prompt = EXPLORER_INSTRUCTION_TEMPLATE.format(
            role="Researcher",
            source_type="academic",
            MIN_SOURCES=5,
            MAX_SOURCES=10
        )
        assert "Prior research context" not in prompt

    def test_prior_context_section_injected(self):
        from src.prompts import build_explorer_instruction
        prompt = build_explorer_instruction(
            role="Researcher",
            source_type="academic",
            min_sources=5,
            max_sources=10,
            prior_context="Key terms: JTMS, belief revision\nConclusion: TMS tracks validity."
        )
        assert "Prior research context" in prompt
        assert "JTMS" in prompt

    def test_prior_context_contains_no_cite_instruction(self):
        from src.prompts import build_explorer_instruction
        prompt = build_explorer_instruction(
            role="Researcher",
            source_type="academic",
            min_sources=5,
            max_sources=10,
            prior_context="Some context"
        )
        assert "do not cite" in prompt.lower() or "must not cite" in prompt.lower()


# ---------------------------------------------------------------------------
# 6. OKF output: directory structure and frontmatter
# ---------------------------------------------------------------------------

class TestOKFOutput:
    def test_write_okf_bundle_creates_directory(self):
        from src.orchestration import write_okf_bundle
        results = {
            "truth-maintenance": "---\ntype: lit-review-topic\ntitle: Truth Maintenance\n---\n\n# Content here"
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            write_okf_bundle(results, tmpdir, research_question="graph-based change-impact analysis")
            assert (Path(tmpdir) / "index.md").exists()
            assert (Path(tmpdir) / "truth-maintenance.md").exists()

    def test_index_md_contains_links_to_all_topics(self):
        from src.orchestration import write_okf_bundle
        results = {
            "truth-maintenance": "---\ntype: lit-review-topic\ntitle: Truth Maintenance\n---\n\nContent",
            "incremental-recomp": "---\ntype: lit-review-topic\ntitle: Incremental Recomputation\n---\n\nContent"
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            write_okf_bundle(results, tmpdir, research_question="graph constraints")
            index_content = (Path(tmpdir) / "index.md").read_text()
            assert "truth-maintenance" in index_content
            assert "incremental-recomp" in index_content

    def test_topic_file_has_okf_frontmatter(self):
        from src.orchestration import write_okf_bundle
        results = {
            "truth-maintenance": "# Truth Maintenance\n\nSome body\n\n## References\n\n1. [Paper](https://arxiv.org/1)"
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            write_okf_bundle(results, tmpdir, research_question="test")
            content = (Path(tmpdir) / "truth-maintenance.md").read_text()
            assert "type: lit-review-topic" in content
            assert "title:" in content
            assert "resource:" in content

    def test_resource_field_points_to_references_anchor(self):
        from src.orchestration import write_okf_bundle
        results = {
            "truth-maintenance": "# Truth Maintenance\n\nBody\n\n## References\n\n1. [P](https://example.com)"
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            write_okf_bundle(results, tmpdir, research_question="test")
            content = (Path(tmpdir) / "truth-maintenance.md").read_text()
            assert "./truth-maintenance.md#references" in content

    def test_wave2_topic_links_to_wave1_dependency(self):
        from src.orchestration import write_okf_bundle
        results = {
            "truth-maintenance": "---\ntype: lit-review-topic\n---\n\n# A",
            "conflict-detection": "---\ntype: lit-review-topic\n---\n\n# C"
        }
        dependencies = {"conflict-detection": ["truth-maintenance"]}
        with tempfile.TemporaryDirectory() as tmpdir:
            write_okf_bundle(results, tmpdir, research_question="test", dependencies=dependencies)
            content = (Path(tmpdir) / "conflict-detection.md").read_text()
            assert "truth-maintenance" in content


# ---------------------------------------------------------------------------
# 7. Planner: produces valid PipelinePlan from topic configs
# ---------------------------------------------------------------------------

class TestPlanner:
    """Planner tests mock the LLM call and verify the output parsing."""
    def test_planner_output_parses_to_pipeline_plan(self):
        from src.orchestration import parse_planner_output, PipelinePlan
        llm_output = '{"wave1": ["a", "b", "d", "e"], "wave2": {"c": ["a"], "f": ["a"], "g": ["a", "c"]}}'
        plan = parse_planner_output(llm_output)
        assert isinstance(plan, PipelinePlan)
        assert "a" in plan.wave1
        assert plan.wave2["c"] == ["a"]

    def test_planner_output_all_topics_covered(self):
        from src.orchestration import parse_planner_output
        llm_output = '{"wave1": ["a", "b"], "wave2": {"c": ["a"]}}'
        plan = parse_planner_output(llm_output)
        all_topics = set(plan.wave1) | set(plan.wave2.keys())
        assert all_topics == {"a", "b", "c"}

    def test_planner_output_with_no_wave2(self):
        from src.orchestration import parse_planner_output
        llm_output = '{"wave1": ["a", "b", "c"], "wave2": {}}'
        plan = parse_planner_output(llm_output)
        assert len(plan.wave1) == 3
        assert len(plan.wave2) == 0

    def test_planner_rejects_invalid_dependency(self):
        from src.orchestration import parse_planner_output
        llm_output = '{"wave1": ["a"], "wave2": {"c": ["z"]}}'
        with pytest.raises(Exception):
            parse_planner_output(llm_output)


# ---------------------------------------------------------------------------
# 8. Distiller: produces DistilledContext from synthesis
# ---------------------------------------------------------------------------

class TestDistiller:
    def test_distiller_output_parses_to_distilled_context(self):
        from src.orchestration import parse_distiller_output, DistilledContext
        llm_output = """{
            "key_terms": ["JTMS", "belief revision"],
            "conclusion": "TMS provides incremental validity tracking.",
            "top_urls": [
                {"title": "JTMS Paper", "url": "https://arxiv.org/123", "source_tier": "peer_reviewed"}
            ]
        }"""
        ctx = parse_distiller_output(llm_output)
        assert isinstance(ctx, DistilledContext)
        assert len(ctx.key_terms) == 2
        assert "JTMS" in ctx.key_terms

    def test_distiller_output_caps_urls_at_5(self):
        from src.orchestration import parse_distiller_output
        urls = [{"title": f"S{i}", "url": f"https://example.com/{i}", "source_tier": "blog_or_forum"} for i in range(6)]
        llm_output = json.dumps({
            "key_terms": ["term"],
            "conclusion": "conclusion",
            "top_urls": urls
        })
        with pytest.raises(Exception):
            parse_distiller_output(llm_output)