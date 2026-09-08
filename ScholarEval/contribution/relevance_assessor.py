#!/usr/bin/env python3
import argparse, json, logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine
from ..engine.codex_transport import CodexError

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--research_plan", required=True)
    parser.add_argument("--papers_file", required=True, help="Extracted papers JSON")
    parser.add_argument("--llm_engine", required=True)
    parser.add_argument("--litellm_name", help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514')")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--cost_log_file", help="Path to centralized cost log file")
    args = parser.parse_args()
    if os.environ.get('SCHOLAREVAL_LLM_BACKEND') == 'codex':
        args.max_workers = int(os.environ.get('SCHOLAREVAL_CODEX_MAX_CONCURRENCY', '1'))

    with open(args.research_plan, "r", encoding='utf-8') as f:
        plan = f.read()
    with open(args.papers_file, "r", encoding="utf-8") as f:
        papers = json.load(f)
    
    llm_cost = model_cost[args.litellm_name] if args.litellm_name else None
    
    cost_lock = threading.Lock()
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    thread_local = threading.local()
    
    def get_thread_llm():
        if not hasattr(thread_local, 'llm'):
            thread_local.llm = LLMEngine(
                llm_engine_name=args.llm_engine, 
                api_key=os.environ.get("API_KEY"), 
                api_endpoint=os.environ.get("API_ENDPOINT")
            )
        return thread_local.llm

    def assess_paper_relevance(paper):
        """Assess relevance of a single paper using thread-local LLM."""
        nonlocal total_cost, total_input_tokens, total_output_tokens
        logging.info(f"Assessing relevance for paper: {paper['paperId']} - {str(paper.get('title') or '')[:60]}")
        llm = get_thread_llm()
        
        prompt = [
            {
            "role": "system",
            "content": (
                "You are an expert at evaluating whether a paper should be considered relevant for assessing the scientific contribution of a research proposal. "
                "You will be given a research proposal and a paper abstract retrieved from semantic scholar. "
                "Your task is to thoroughly understand the scientific contributions of each of the research proposal and the paper, and to output a score from 0 to 5 indicating how similar the contributions of the paper are to those of the research proposal. "
                
                "**Scoring Rubric:**\n\n"
                
                "**Score 5 - Highly Similar Contributions:**\n"
                "- The paper addresses the exact same research question or hypothesis as the proposal\n"
                "- Uses identical or very similar methodological approaches\n"
                "- Targets the same specific population, system, or domain\n"
                "- Would directly compete with or overlap significantly with the proposed research\n"
                "- The paper's findings would substantially impact the novelty of the proposed work\n\n"
                
                "**Score 4 - Very Similar Contributions:**\n"
                "- The paper addresses a closely related research question with significant overlap\n"
                "- Uses similar methodological approaches with minor variations\n"
                "- Targets a very similar population, system, or domain\n"
                "- Shares most key variables, measurements, or outcomes of interest\n"
                "- The paper's contributions would moderately impact the proposed research's novelty\n\n"
                
                "**Score 3 - Moderately Similar Contributions:**\n"
                "- The paper addresses a related research question within the same broad area\n"
                "- Uses some similar methods or approaches but with notable differences\n"
                "- Targets a related but distinct population, system, or domain\n"
                "- Shares some key concepts, variables, or theoretical frameworks\n"
                "- The paper provides useful context but doesn't directly threaten novelty\n\n"
                
                "**Score 2 - Somewhat Similar Contributions:**\n"
                "- The paper is in the same general field or discipline\n"
                "- Uses different methods but addresses conceptually related problems\n"
                "- Limited overlap in specific research focus or target populations\n"
                "- Shares broad theoretical background but differs in specific contributions\n"
                "- The paper is peripherally relevant for background or context\n\n"
                
                "**Score 1 - Minimally Similar Contributions:**\n"
                "- The paper is tangentially related to the research area\n"
                "- Very limited overlap in methods, populations, or specific research questions\n"
                "- May share some terminology or broad field classification\n"
                "- Provides minimal insight relevant to the proposed research\n"
                "- Connection is primarily at the disciplinary level\n\n"
                
                "**Score 0 - No Similar Contributions:**\n"
                "- The paper addresses completely different research questions\n"
                "- No meaningful overlap in methods, populations, or domains\n"
                "- Different field or discipline entirely\n"
                "- No relevant insights for the proposed research\n"
                "- No discernible connection between the contributions\n\n"
                
                "**Instructions:** Carefully analyze both the research proposal and paper abstract, identify the core scientific contributions of each, and assign the most appropriate score based on the degree of similarity in their contributions."
                )
            },
            {
                "role": "user",
                "content": (
                f"Research proposal:\n{plan}\n\n"
                f"Paper abstract:\n{paper.get('abstract')}\n\n"
                "Question: How similar are this paper's contributions to those of the research proposal? "
                "Respond with a JSON object containing two fields 'rationale' and 'score'. Please output a parseable JSON block as follows, being especially careful to use the correct number of escape characters:\n```json\n{{\n  \"rationale\": <your detailed reasoning for the score>, \"score\": <an integer from 0 to 5 based on the rubric>]\n}}\n```"
                )
            }
        ]

        resp, input_tokens, output_tokens = llm.respond(prompt, temperature=0)
        
        with cost_lock:
            if args.litellm_name:
                if args.litellm_name == "meta_llama/Llama-3.3-70B-Instruct":
                    cost = 0
                else:
                    cost = (llm_cost["input_cost_per_token"] * input_tokens + 
                            llm_cost["output_cost_per_token"] * output_tokens)
            else:
                cost = 0
            total_cost += cost
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
        
        try:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', resp, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*?\}', resp, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    raise ValueError("No JSON found in response")
            
            assessment = json.loads(json_str)
            
            paper_with_assessment = paper.copy()
            paper_with_assessment['relevance_score'] = assessment.get('score')
            paper_with_assessment['relevance_rationale'] = assessment.get('rationale')
            paper_with_assessment['cost'] = cost
            paper_with_assessment['input_tokens'] = input_tokens
            paper_with_assessment['output_tokens'] = output_tokens
            
            logging.info(f"  -> Score: {assessment.get('score', 0)}")
            return paper_with_assessment
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            from ScholarEval.utils.workflow_errors import ScientificValidationError
            raise ScientificValidationError('Relevance response could not be parsed; no score fabricated') from e
    # Filter out papers with null abstracts before processing
    papers_with_abstracts = [paper for paper in papers if paper.get('abstract')]
    logging.info(f"Filtered out {len(papers) - len(papers_with_abstracts)} papers with null abstracts")
    
    # Separate papers that already have relevance scores from those that need processing
    already_scored = [paper for paper in papers_with_abstracts if 'relevance_score' in paper]
    needs_scoring = [paper for paper in papers_with_abstracts if 'relevance_score' not in paper]
    
    logging.info(f"Found {len(already_scored)} papers already scored, {len(needs_scoring)} papers need scoring")
    
    # Process papers that need scoring in parallel
    relevant = already_scored.copy()  # Start with already scored papers

    from ScholarEval.utils.durable import ItemStore
    from ScholarEval.utils.checkpoints import valid_relevance
    def durable_assess(paper):
        return ItemStore.current('relevance').run([plan, paper],
            lambda: assess_paper_relevance(paper), valid_relevance)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_paper = {
            executor.submit(durable_assess, paper): paper
            for paper in needs_scoring
        }
        
        for future in as_completed(future_to_paper):
            try:
                paper_with_assessment = future.result()
                relevant.append(paper_with_assessment)
            except Exception:
                for pending in future_to_paper:
                    pending.cancel()
                raise
    # Stable paper ordering prevents scheduling from changing sampling ties.
    order = {p['paperId']: i for i, p in enumerate(papers_with_abstracts)}
    relevant.sort(key=lambda p: order[p['paperId']])
    print(f"Total LLM cost for contribution_relevance_assessor: ${total_cost:.6f}")
    logging.info(f"{len(relevant)} papers assessed with relevance scores.")
    
    # Log cost to centralized cost log if provided
    if args.cost_log_file:
        cost_entry = {
            "step": "contribution_relevance_assessor",
            "cost": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
        with open(args.cost_log_file, 'a', encoding='utf-8') as f:
            json.dump(cost_entry, f)
            f.write('\n')
    
    # Save output without cost summary
    output_data = {
        "papers": relevant
    }
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
if __name__ == "__main__":
    from ScholarEval.utils.checkpoints import checked_main
    checked_main(main, "ScholarEval.contribution.relevance_assessor")
