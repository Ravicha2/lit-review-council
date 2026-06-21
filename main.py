import argparse
import asyncio
from src.orchestration import orchestrate

async def main():
    parser = argparse.ArgumentParser(description="Run the Literature Review Orchestrator Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the topics YAML configuration file")
    parser.add_argument("--output", type=str, default="okf_output", help="Output directory for the OKF bundle (default: okf_output)")
    parser.add_argument("--question", type=str, default="General Literature Review", help="The overarching research question")
    
    args = parser.parse_args()
    
    print(f"Starting orchestration with config: {args.config}")
    print(f"Output directory: {args.output}")
    print(f"Research Question: {args.question}")
    print("-" * 50)
    
    await orchestrate(
        config_path=args.config, 
        output_dir=args.output, 
        research_question=args.question
    )

if __name__ == "__main__":
    asyncio.run(main())
