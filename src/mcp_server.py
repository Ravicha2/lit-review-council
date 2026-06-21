import os
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import List, Optional

from src.orchestration import orchestrate, Config, TopicConfig

mcp = FastMCP("lit-review-council")

@mcp.tool()
async def conduct_literature_review(question: str, topics: list[dict], output_dir: str = ".") -> str:
    """
    Executes a multi-agent literature review on the provided research question and topics.
    
    Args:
        question: The overarching research question driving the literature review.
        topics: A list of topic dictionaries. Each topic must have:
            - slug: (str) a short hyphenated identifier
            - description: (str) explanation of the topic
            - search_keywords: (list[str]) 2-4 highly specific search queries
            - rationale: (str, optional) why this topic was chosen
        output_dir: (str) Directory to save the final OKF markdown bundle. Defaults to current directory.
        
    Returns:
        A success message with the path to the generated OKF bundle directory.
    """
    # Convert dicts to TopicConfig objects
    topic_configs = []
    for t in topics:
        topic_configs.append(TopicConfig(
            slug=t["slug"],
            description=t["description"],
            search_keywords=t["search_keywords"],
            rationale=t.get("rationale")
        ))
        
    config = Config(topics=topic_configs)
    
    print(f"Starting literature review for: {question}")
    print(f"Output directory: {output_dir}")
    
    await orchestrate(
        config=config,
        output_dir=output_dir,
        research_question=question
    )
    
    index_path = os.path.join(output_dir, "index.md")
    return f"Literature review completed successfully. The OKF bundle has been saved to {output_dir}. Start reading at {index_path}."

if __name__ == "__main__":
    mcp.run()
