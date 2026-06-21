import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import yaml
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Optional, Any

from src.schema import Reference
from src.pipeline import run_pipeline, eng_model, research_model
from google.genai.types import Content, Part

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TopicConfig(BaseModel):
    slug: str
    description: str
    search_keywords: List[str]
    rationale: Optional[str] = None

class Config(BaseModel):
    topics: List[TopicConfig]

class PipelinePlan(BaseModel):
    wave1: List[str]
    wave2: Dict[str, List[str]]

    @model_validator(mode='after')
    def validate_plan(self):
        wave1_set = set(self.wave1)
        wave2_set = set(self.wave2.keys())
        
        # no topic in both waves
        if wave1_set.intersection(wave2_set):
            raise ValueError("A topic cannot be in both wave1 and wave2")
            
        for deps in self.wave2.values():
            if len(deps) > 3:
                raise ValueError("Wave 2 topics can have a maximum of 3 dependencies")
            for dep in deps:
                if dep not in wave1_set and dep not in wave2_set:
                    raise ValueError(f"Dependency '{dep}' must be a wave 1 topic")
        return self

class DistilledContext(BaseModel):
    key_terms: List[str]
    conclusion: str
    top_urls: List[Reference]

    @model_validator(mode='after')
    def validate_context(self):
        if len(self.top_urls) > 5:
            raise ValueError("top_urls cannot exceed 5 items")
        return self

class ExecutionPlan(BaseModel):
    wave1_topics: List[str]
    wave2_contexts: Dict[str, List[str]]

# ---------------------------------------------------------------------------
# Orchestration Logic
# ---------------------------------------------------------------------------

def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)

def parse_planner_output(llm_output: str) -> PipelinePlan:
    # Handle possible markdown backticks
    if llm_output.startswith("```json"):
        llm_output = llm_output[7:]
    if llm_output.endswith("```"):
        llm_output = llm_output[:-3]
    return PipelinePlan.model_validate_json(llm_output)

def parse_distiller_output(llm_output: str) -> DistilledContext:
    if llm_output.startswith("```json"):
        llm_output = llm_output[7:]
    if llm_output.endswith("```"):
        llm_output = llm_output[:-3]
    return DistilledContext.model_validate_json(llm_output)

def handle_wave1_failure(topic_slug: str, error: Exception) -> DistilledContext:
    # Returns empty distillation on failure
    return DistilledContext(
        key_terms=[],
        conclusion="",
        top_urls=[]
    )

def build_execution_plan(plan: PipelinePlan) -> ExecutionPlan:
    wave1_topics = plan.wave1
    wave2_contexts = {}
    
    wave1_set = set(plan.wave1)
    
    for topic, deps in plan.wave2.items():
        # Only wave 1 topics can be provided as prior context
        valid_deps = [dep for dep in deps if dep in wave1_set]
        wave2_contexts[topic] = valid_deps
        
    return ExecutionPlan(wave1_topics=wave1_topics, wave2_contexts=wave2_contexts)

def build_prior_context_string(dependencies: List[str], distillations: Dict[str, DistilledContext]) -> str:
    parts = []
    # Maximum 3 distillations
    count = 0
    for dep in dependencies:
        if count >= 3:
            break
        dist = distillations.get(dep)
        if dist is None:
            continue
        
        # Check if the distillation is empty (from failure)
        if not dist.key_terms and not dist.conclusion:
            continue
            
        part = f"Topic: {dep}\nKey Terms: {', '.join(dist.key_terms)}\nConclusion: {dist.conclusion}\nTop Sources:"
        for url in dist.top_urls:
            part += f"\n- [{url.title}]({url.url})"
        parts.append(part)
        count += 1
        
    return "\n\n".join(parts)

def write_okf_bundle(results: Dict[str, str], output_dir: str, research_question: str, dependencies: Optional[Dict[str, List[str]]] = None):
    os.makedirs(output_dir, exist_ok=True)
    if dependencies is None:
        dependencies = {}
        
    # Write index.md
    index_path = os.path.join(output_dir, "index.md")
    with open(index_path, "w") as f:
        f.write(f"# Literature Review: {research_question}\n\n## Topics\n")
        for slug in results.keys():
            f.write(f"- [{slug}](./{slug}.md)\n")
            
    # Write topic files
    for slug, markdown in results.items():
        # Make sure frontmatter exists, else inject a basic one
        if not markdown.strip().startswith("---"):
            frontmatter = f"---\ntype: lit-review-topic\ntitle: {slug}\nresource: ./{slug}.md#references\n---\n\n"
            markdown = frontmatter + markdown
        else:
            # Inject resource field if it doesn't exist
            lines = markdown.split("\n")
            if len(lines) > 1 and lines[0] == "---":
                end_frontmatter = -1
                for i in range(1, len(lines)):
                    if lines[i] == "---":
                        end_frontmatter = i
                        break
                if end_frontmatter != -1:
                    has_resource = any(line.startswith("resource:") for line in lines[1:end_frontmatter])
                    has_type = any(line.startswith("type:") for line in lines[1:end_frontmatter])
                    
                    inject_lines = []
                    if not has_type:
                        inject_lines.append("type: lit-review-topic")
                    if not has_resource:
                        inject_lines.append(f"resource: ./{slug}.md#references")
                        
                    lines = lines[:end_frontmatter] + inject_lines + lines[end_frontmatter:]
                    markdown = "\n".join(lines)
            
        # Append links to dependencies if it's a wave 2 topic
        if slug in dependencies and dependencies[slug]:
            markdown += "\n\n## Dependencies\n"
            for dep in dependencies[slug]:
                markdown += f"- [{dep}](./{dep}.md)\n"
                
        topic_path = os.path.join(output_dir, f"{slug}.md")
        with open(topic_path, "w") as f:
            f.write(markdown)

# ---------------------------------------------------------------------------
# LLM Functions
# ---------------------------------------------------------------------------

async def run_planner(topics: List[TopicConfig]) -> PipelinePlan:
    prompt = f"""You are an expert technical orchestrator. Analyze the following {len(topics)} topics for a literature review:
    
"""
    for t in topics:
        prompt += f"- Slug: {t.slug}\n  Description: {t.description}\n  Rationale: {t.rationale}\n\n"
        
    prompt += """Organize these topics into exactly two waves.
- Wave 1 topics run in parallel and should be foundational concepts.
- Wave 2 topics run after Wave 1, and each can depend on up to 3 Wave 1 topics.
Output a JSON object matching this schema:
{
    "wave1": ["topic-slug-1", "topic-slug-2"],
    "wave2": {
        "wave2-slug-1": ["wave1-dep-1"],
        "wave2-slug-2": ["wave1-dep-1", "wave1-dep-2"]
    }
}
Do not include markdown backticks. Just output the JSON.
"""
    # Assuming eng_model from pipeline is available and has an async completion method.
    # Wait, LiteLlm has a completion method? We can use run() but ADK's LiteLlm is a model.
    # We can just send a request using the provider directly, or if eng_model has predict().
    # Actually, let's look at how agents use it. We'll use eng_model.predict if it exists, or just use the agent framework?
    # The tests just mock `parse_planner_output`. I will define these as dummy for now or standard ADK model calling.
    
    try:
        response = await eng_model.predict_async(prompt)
        # Handle whatever response format LiteLlm returns
        text = response.text if hasattr(response, "text") else str(response)
    except AttributeError:
        # Fallback if predict_async is not the method
        # Using litellm directly
        import litellm
        response = await litellm.acompletion(
            model=os.getenv("ENG_MODEL", "openrouter/deepseek/deepseek-v4-flash"),
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content
        
    return parse_planner_output(text)

async def run_distiller(topic: str, markdown: str) -> DistilledContext:
    prompt = f"""You are an expert synthesist. Read the following literature review report for the topic '{topic}' and distill it into a JSON object.

Report:
{markdown}

Output a JSON object matching this schema:
{{
    "key_terms": ["list", "of", "important", "jargon"],
    "conclusion": "A 1-2 sentence high-level conclusion.",
    "top_urls": [
        {{ "title": "Source Title", "url": "https://example.com", "source_tier": "peer_reviewed | established_project | vendor_doc | blog_or_forum" }}
    ]
}}
Ensure top_urls contains a maximum of 5 references. Do not include markdown backticks. Just output the JSON.
"""
    try:
        import litellm
        response = await litellm.acompletion(
            model=os.getenv("ENG_MODEL", "openrouter/deepseek/deepseek-v4-flash"),
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content
    except Exception as e:
        print(f"Error in distiller: {e}")
        text = '{"key_terms": [], "conclusion": "", "top_urls": []}'
        
    return parse_distiller_output(text)

async def orchestrate(config_path: str, output_dir: str, research_question: str):
    config = load_config(config_path)
    
    print("Running planner...")
    plan = await run_planner(config.topics)
    exec_plan = build_execution_plan(plan)
    
    print(f"Wave 1 topics: {exec_plan.wave1_topics}")
    
    async def run_and_distill(slug: str):
        try:
            markdown = await run_pipeline(slug, session_id=slug, output_dir=output_dir, prior_context=None)
            if markdown is None:
                raise ValueError(f"Pipeline returned None for topic {slug}")
            distillation = await run_distiller(slug, markdown)
            return slug, markdown, distillation
        except Exception as e:
            print(f"Topic {slug} failed: {e}")
            return slug, None, handle_wave1_failure(slug, e)
            
    wave1_tasks = [run_and_distill(slug) for slug in exec_plan.wave1_topics]
    wave1_results = await asyncio.gather(*wave1_tasks)
    
    results_markdown = {}
    distillations = {}
    
    for slug, markdown, dist in wave1_results:
        if markdown:
            results_markdown[slug] = markdown
        distillations[slug] = dist
        
    print(f"Wave 2 topics: {list(exec_plan.wave2_contexts.keys())}")
    
    async def run_wave2(slug: str, deps: List[str]):
        prior_context = build_prior_context_string(deps, distillations)
        try:
            markdown = await run_pipeline(slug, session_id=slug, output_dir=output_dir, prior_context=prior_context)
            if markdown:
                return slug, markdown
        except Exception as e:
            print(f"Topic {slug} failed: {e}")
        return slug, None

    wave2_tasks = [run_wave2(slug, deps) for slug, deps in exec_plan.wave2_contexts.items()]
    wave2_results = await asyncio.gather(*wave2_tasks)
    
    for slug, markdown in wave2_results:
        if markdown:
            results_markdown[slug] = markdown
            
    print(f"Writing OKF bundle to {output_dir}...")
    write_okf_bundle(results_markdown, output_dir, research_question, dependencies=plan.wave2)
    print("Orchestration complete.")
