#!/usr/bin/env python3
import argparse
import json
import logging
import time
import os
import glob
import subprocess
import asyncio
import sys
from ..utils.checkpoints import StageCheckpoint, digest, atomic_json
from ..utils.retrieval_progress import RetrievalProgress
from datetime import datetime
from ..utils.grobid import GrobidService, GrobidStartupError, parse_pdf_corpus
from ..utils.durable import ItemStore
from pathlib import Path
from ..utils.retrieval_http import RetrievalError, RetrievalResponseError
from ..utils.semantic_scholar import SemanticScholar
from ..utils.string_utils import GrobidXMLParser
from ..utils.pdf_utils import FastPDFDownloader

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def is_date_after(cutoff: str, paper: str) -> bool:
    if not cutoff:
        return True
    if not paper:
        logging.warning('Skipping undated paper under cutoff policy')
        return False
    cutoff_date = datetime.strptime(cutoff, "%Y-%m-%d")
    try:
        paper_date = datetime.strptime(paper, "%Y-%m-%d")
    except (TypeError, ValueError):
        logging.warning('Skipping paper with uninterpretable publication date under cutoff policy')
        return False
    return cutoff_date > paper_date

async def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--relevant_papers", required=True, help="JSON of relevant papers")
    parser.add_argument("--output_file", required=True, help="Output JSON file")
    parser.add_argument("--rec_limit", type=int, default=8, help="Number of recs per paper")
    parser.add_argument("--sleep_between_calls", type=float, default=1,
                        help="Seconds to sleep between API calls")
    parser.add_argument("--pdf_dir", default="./pdfs", help="Directory to store downloaded PDFs")
    parser.add_argument("--grobid_config", default="./GROBID_config.json", help="GROBID config file path")
    parser.add_argument("--augmentation_type", choices=["related_work", "all"], default="all",
                        help="Type of augmentation: 'related_work' (recommendations + related work sections) or 'all' (recommendations + all references)")
    parser.add_argument("--cutoff_date", help="Optional cutoff date (YYYY-MM-DD) for literature search.")
    parser.add_argument("--max_refs_per_paper", type=int, default=10,
                        help="Max number of references to fetch per paper (used with 'all' augmentation type)")
    args = parser.parse_args()

    s2 = SemanticScholar(api_key=os.environ.get("S2_API_KEY"))
    checkpoint = StageCheckpoint('ScholarEval.contribution.paper_augmentation', sys.argv[1:])
    primary = os.environ.get('SCHOLAREVAL_STAGE_PRIMARY', args.output_file)
    progress = RetrievalProgress(primary + '.progress.json', checkpoint.fingerprint,
        s2.http, resume=os.environ.get('SCHOLAREVAL_RESUME') == '1')

    def cached(method, *values, **kwargs):
        return progress.run(digest([method, values, kwargs]), lambda: getattr(s2, method)(*values, **kwargs))

    downloader = FastPDFDownloader()
    
    papers_data = json.loads(Path(args.relevant_papers).read_text(encoding='utf-8'))
    papers = papers_data['papers'] if 'papers' in papers_data else papers_data
    augmented = {p['paperId']: p for p in papers}
    initial_paper_count = len(papers)
    
    recommendations_added = 0
    related_work_added = 0
    references_added = 0  
    pdfs_downloaded = 0
    downloaded_paths = []
    xml_files_processed = 0

    # Filter papers with relevance score >= 3
    high_relevance_papers = [p for p in papers if p.get('relevance_score', 0) >= 3]
    logging.info(f"Starting with {initial_paper_count} papers")
    logging.info(f"Processing {len(high_relevance_papers)} papers with relevance score >= 3")
    logging.info(f"Augmentation type: {args.augmentation_type}")

    for index, p in enumerate(high_relevance_papers, 1):
        pid = p['paperId']
        relevance_score = p.get('relevance_score', 0)
        logging.info(f"Augmentation {index}/{len(high_relevance_papers)}: {pid} (relevance score: {relevance_score})")

        logging.info(f" → Fetching recommendations (limit={args.rec_limit})")
        recommendations_for_this_paper = 0
        try:
            recs = cached('get_recommendations_multi_seed', [pid], limit=args.rec_limit)
            for r in recs:
                rid = r['paperId']
                logging.info(f"    • Recommendation: {rid}")
                if rid not in augmented:
                    # The multi-seed Recommendations API already returns the
                    # complete metadata requested by this pipeline. Reuse it
                    # instead of issuing one extra Graph API request per paper.
                    paper_details = r
                    if (is_date_after(args.cutoff_date, paper_details.get('publicationDate', '')) if args.cutoff_date else True):
                        augmented[rid] = paper_details
                        recommendations_added += 1
                        recommendations_for_this_paper += 1
            logging.info(f"   Added {recommendations_for_this_paper} new recommendations for {pid}")
        except (RetrievalError, GrobidStartupError):
            raise
        except Exception as error:
            raise
        # All calls are paced by the shared S2 policy, including cache misses.

        if args.augmentation_type == "related_work":
            # Extract from related work sections in PDFs
            logging.info(f" → Processing related work section for {pid}")
        else:  # augmentation_type == "all"
            # Use simple references API (from old code)
            logging.info(f" → Fetching all references (max={args.max_refs_per_paper or 'all'})")
        try:
            if args.augmentation_type == "related_work":
                # Get paper metadata to find PDF
                paper_details = p
                pdf_url = None
                
                if paper_details.get('openAccessPdf'):
                    pdf_info = paper_details['openAccessPdf']
                    if pdf_info.get('url'):
                        pdf_url = pdf_info['url']
                    elif pdf_info.get('disclaimer'):
                        pdf_url = downloader.extract_url(pdf_info['disclaimer'])
                
                if not pdf_url:
                    logging.warning(f"No PDF available for {pid}, skipping related work extraction")
                    continue
                    
                # Download PDF
                corpus_id = str(paper_details.get('corpusId', pid))
                pdf_path = await download_single_pdf(downloader, pdf_url, corpus_id, args.pdf_dir)
                
                if not pdf_path:
                    if os.environ.get('SCHOLAREVAL_FULL_TEXT_POLICY', 'available-evidence') == 'available-evidence':
                        logging.warning('PDF unavailable for %s; retaining metadata evidence', pid)
                        continue
                    raise RetrievalError(f'Full text required but unavailable for {pid}')
                    
                # Process with GROBID (defer to batch processing after all downloads)
                pdfs_downloaded += 1
                downloaded_paths.append(pdf_path)
                logging.info(f"PDF downloaded successfully for {pid} (total PDFs: {pdfs_downloaded})")
                
            else:  # augmentation_type == "all"
                # Use simple references API (from old code)
                refs = cached('get_references', pid, max_references=args.max_refs_per_paper)
                references_for_this_paper = 0
                for cited in refs:
                    cid = cited.get("paperId")
                    if not cid:
                        continue
                    logging.info(f"    • Reference: {cid}")
                    if cid not in augmented:
                        # get_references() already asks S2 for the same core
                        # metadata fields as get_paper_details(). Reuse cited
                        # metadata and avoid one extra request per reference.
                        if is_date_after(args.cutoff_date, cited.get('publicationDate')):
                            augmented[cid] = cited
                            references_added += 1
                        references_for_this_paper += 1
                logging.info(f"   Added {references_for_this_paper} new references for {pid}")
                
        except (RetrievalError, GrobidStartupError):
            raise
        except Exception as error:
            raise
        

    # Process all downloaded PDFs with GROBID in batch (only for related_work augmentation)
    if args.augmentation_type == "related_work" and pdfs_downloaded > 0:
        logging.info(f"Processing {pdfs_downloaded} downloaded PDFs with GROBID...")
        try:
            parse_path = Path(primary + '.parse-progress.json')
            try:
                parse_records = json.loads(parse_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                parse_records = {}
            from ..utils.checkpoints import file_hash
            def parsed(cid, record):
                parse_records[cid] = record
                atomic_json(parse_path, parse_records)
            pending = []
            for pdf_path in downloaded_paths:
                pdf = Path(pdf_path)
                xml = pdf.with_suffix('.grobid.tei.xml')
                old = parse_records.get(pdf.stem, {})
                if not (old.get('status') == 'parsed' and old.get('pdf_sha256') == file_hash(pdf)
                        and xml.exists() and old.get('xml_sha256') == file_hash(xml)):
                    pending.append(pdf)
            parse_pdf_corpus(pending, args.pdf_dir, parsed, config=args.grobid_config, metrics=progress.state['metrics'])
            pdf_files = downloaded_paths
            for pdf_path in pdf_files:
                # GROBID creates .grobid.tei.xml files, not .xml
                xml_path = pdf_path.replace('.pdf', '.grobid.tei.xml')
                if os.path.exists(xml_path):
                    xml_files_processed += 1
                    try:
                        corpus_id = os.path.basename(pdf_path).replace('.pdf', '')
                        logging.info(f"[{xml_files_processed}/{len(pdf_files)}] Extracting related work from {corpus_id}")
                        
                        logging.info(f"   Calling extract_related_work_references for {corpus_id}")
                        related_work_refs = ItemStore.current('related_work').run(file_hash(xml_path),
                            lambda: extract_related_work_references(xml_path, s2))
                        logging.info(f"   extract_related_work_references returned {len(related_work_refs) if related_work_refs else 0} papers")
                        
                        # Add related work papers to augmented list
                        new_papers_from_this_xml = 0
                        for ref_paper in related_work_refs:
                            ref_id = ref_paper.get('paperId')
                            if ref_id and ref_id not in augmented and is_date_after(args.cutoff_date, ref_paper.get('publicationDate')):
                                logging.info(f"    • Added related work paper: {ref_id}")
                                augmented[ref_id] = ref_paper
                                related_work_added += 1
                                new_papers_from_this_xml += 1
                            elif ref_id:
                                logging.info(f"    • Skipped duplicate paper: {ref_id}")
                                
                        logging.info(f"   Found {len(related_work_refs)} total references, added {new_papers_from_this_xml} new papers from {corpus_id}")
                        
                    except (RetrievalError, GrobidStartupError):
                        raise
                    except Exception as error:
                        raise
                else:
                    if os.environ.get('SCHOLAREVAL_FULL_TEXT_POLICY', 'available-evidence') != 'available-evidence':
                        raise RetrievalError(f'Full text required but parse failed for {pdf_path}')
                    logging.warning('Parse unavailable for %s; retaining metadata evidence', pdf_path)
            
            logging.info(f"Processed {xml_files_processed} XML files out of {len(pdf_files)} PDFs")
            
        except (RetrievalError, GrobidStartupError):
            raise
        except Exception as error:
            raise
    else:
        if args.augmentation_type == "related_work":
            logging.info("No PDFs downloaded for related work extraction")
        else:
            logging.info("Skipping GROBID processing for 'all' augmentation type")

    progress.state['status'] = 'completed'
    progress.state['metrics'].update(pdfs_downloaded=pdfs_downloaded, pdfs_parsed=xml_files_processed,
                                     unique_papers_found=len(augmented))
    progress.save()
    atomic_json(args.output_file + '.metrics.json', {'retrieval': progress.state['metrics']})
    # Save augmented results
    out = list(augmented.values())
    final_paper_count = len(out)
    
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    
    # Summary statistics
    logging.info("=" * 60)
    logging.info("AUGMENTATION SUMMARY:")
    logging.info(f"Initial papers: {initial_paper_count}")
    logging.info(f"Papers processed (relevance >= 3): {len(high_relevance_papers)}")
    logging.info(f"PDFs downloaded: {pdfs_downloaded}")
    logging.info(f"XML files processed: {xml_files_processed}")
    logging.info(f"Papers added via recommendations: {recommendations_added}")
    if args.augmentation_type == "related_work":
        logging.info(f"Papers added via related work: {related_work_added}")
        total_added = recommendations_added + related_work_added
    else:
        logging.info(f"Papers added via references: {references_added}")
        total_added = recommendations_added + references_added
    logging.info(f"Total papers added: {total_added}")
    logging.info(f"Final paper count: {final_paper_count}")
    logging.info(f"Growth factor: {final_paper_count/max(initial_paper_count, 1):.2f}x")
    logging.info("=" * 60)
    logging.info(f"Saved {final_paper_count} papers to {args.output_file}")

async def download_single_pdf(downloader, pdf_url, corpus_id, pdf_dir):
    """Download a single PDF and return the file path."""
    try:
        # Use the async downloader for a single PDF
        pdf_data = [(pdf_url, corpus_id)]
        results = await downloader.download_pdfs_batch_async(pdf_data, save_dir=pdf_dir)
        
        if results and results[0] and not isinstance(results[0], Exception):
            return os.path.join(pdf_dir, f"{corpus_id}.pdf")
        return None
    except (RetrievalError, GrobidStartupError):
        raise
    except Exception as error:
        raise

def extract_related_work_references(xml_path, s2_client):
    """Extract papers referenced in the related work section."""
    try:
        with open(xml_path, "r", encoding="utf-8") as file:
            xml_content = file.read()
            
        parser = GrobidXMLParser(xml_content)
        
        # Find related work section
        related_work_section = parser.find_related_work_section()
        
        if not related_work_section:
            logging.info("   No related work section found")
            return []
        
        logging.info(f"   Found related work section: '{related_work_section['header']}'")
        logging.info(f"   Related work section length: {len(related_work_section['full_text'])} characters")
        
        # Get bibliography entries
        bibliography = parser.extract_bibliography()
        
        if not bibliography:
            logging.info("   No bibliography found")
            return []
        
        # Extract reference citations from related work text
        section_refs = parser.extract_references_from_section(related_work_section['full_text'])
        
        logging.info(f"   Found {len(bibliography)} total bibliography entries")
        logging.info(f"   Found {len(section_refs)} reference citations in related work section")
        
        # Filter bibliography to only entries mentioned in related work
        relevant_bib_entries = []
        for bib_entry in bibliography:
            if any(ref_id in section_refs for ref_id in [bib_entry.get('xml:id', ''), bib_entry.get('id', '')]):
                relevant_bib_entries.append(bib_entry)
        
        logging.info(f"   {len(relevant_bib_entries)} bibliography entries are referenced in related work section")
        
        # Search for papers in bibliography using Semantic Scholar
        found_papers = []
        search_limit = min(20, len(relevant_bib_entries))  # Use relevant entries, limit to 20
        entries_to_search = relevant_bib_entries if relevant_bib_entries else bibliography[:20]
        
        logging.info(f"   Searching for {search_limit} papers from {'relevant' if relevant_bib_entries else 'first 20'} bibliography entries...")
        
        for i, bib_entry in enumerate(entries_to_search[:search_limit]):
            if not bib_entry.get('title') or not bib_entry.get('authors'):
                continue
                
            title = bib_entry['title'].strip()
            authors = bib_entry['authors']
            
            if len(title) < 10:  # Skip very short titles
                continue
                
            logging.info(f"   [{i+1}/{search_limit}] Searching for: {title[:50]}...")
            
            try:
                # Search using title and first author
                query = title
                if authors and len(authors) > 0:
                    first_author = authors[0].split()[-1]  # Get last name
                    query = f"{title} {first_author}"
                
                # Use the existing search method
                search_results = s2_client.search_top_papers(query, limit=3)
                
                if search_results:
                    # Find best match based on title similarity
                    best_match = find_best_title_match(title, search_results)
                    
                    if best_match:
                        # Get full paper details
                        paper_details = s2_client.get_paper_details(best_match['paperId'])
                        found_papers.append(paper_details)
                        logging.info(f"     ✓ Found: {best_match['title'][:50]}...")
                    else:
                        logging.info(f"     ✗ No good match found (similarity too low)")
                else:
                    logging.info(f"     ✗ No search results from Semantic Scholar")
                    
            except (RetrievalError, GrobidStartupError):
                raise
            except Exception as error:
                raise
            
            # Rate limiting
            time.sleep(1)
            
        logging.info(f"   Successfully found {len(found_papers)} papers from related work section")
        
        return found_papers
        
    except (RetrievalError, GrobidStartupError):
        raise
    except Exception as error:
        raise

def find_best_title_match(target_title, search_results):
    """Find the best matching paper based on title similarity."""
    target_words = set(target_title.lower().split())
    best_match = None
    best_score = 0
    
    for paper in search_results:
        if not paper.get('title'):
            continue
            
        paper_title = paper['title'].lower()
        paper_words = set(paper_title.split())
        
        intersection = len(target_words & paper_words)
        union = len(target_words | paper_words)
        
        if union > 0:
            similarity = intersection / union
            
            substring_bonus = 0
            if target_title.lower() in paper_title or paper_title in target_title.lower():
                substring_bonus = 0.2
            
            total_score = similarity + substring_bonus
            
            if total_score > best_score and total_score > 0.5: 
                best_score = total_score
                best_match = paper
    
    return best_match

if __name__ == "__main__":
    from ScholarEval.utils.checkpoints import checked_main
    checked_main(lambda: asyncio.run(main()), "ScholarEval.contribution.paper_augmentation")
