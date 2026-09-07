"""Offline partial coverage regressions: fake S2, PDF transport and GROBID only."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import requests
from PyPDF2 import PdfWriter

from ScholarEval.soundness.snippet_search import retrieve
from ScholarEval.utils.checkpoints import StageCheckpoint, atomic_json
from ScholarEval.utils.grobid import GrobidStartupError, parse_pdf_corpus
from ScholarEval.utils.pdf_utils import FastPDFDownloader
from ScholarEval.utils.retrieval_http import RetrievalNetworkError, RetrievalResponseError
from ScholarEval.utils.retrieval_progress import RetrievalProgress
from ScholarEval.utils.semantic_scholar import SemanticScholar


TEI = '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body><div><head n="1">Methods</head><p>Observed experimental evidence.</p></div></body></text></TEI>'


def write_pdf(path):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open('wb') as stream:
        writer.write(stream)


class PartialCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pdfs = self.root / 'pdfs'
        self.pdfs.mkdir()
        self.args = SimpleNamespace(pdf_dir=str(self.pdfs), queries_file=str(self.root / 'q.json'),
            methods_file=str(self.root / 'm.json'), output_file=str(self.root / 'snippet'),
            cutoff_date=None, research_title='', research_abstract='')
        atomic_json(self.args.methods_file, {'clean_methods': ['method']})
        atomic_json(self.args.queries_file, {'queries': {'method': 'query'}})
        self.s2 = SemanticScholar(None)
        self.progress = RetrievalProgress(self.root / 'progress.json', 'fixture', self.s2.http)
        self.service = Mock(metrics={'grobid_action': 'reused', 'grobid_startup_seconds': 0})
        self.service.healthy.return_value = True
        self.service.diagnostics.return_value = 'Fixture service unavailable'
        self.enterContext(patch.dict(os.environ, {'SCHOLAREVAL_FULL_TEXT_POLICY': 'available-evidence',
                                                'SCHOLAREVAL_PDF_MAX_ATTEMPTS': '1'}))
        self.enterContext(patch('ScholarEval.utils.grobid.GrobidService', return_value=self.service))
        self.enterContext(patch('grobid_client.grobid_client.GrobidClient.__init__', return_value=None))
        self.process = self.enterContext(patch('grobid_client.grobid_client.GrobidClient.process_pdf',
                                              side_effect=lambda service, pdf: (pdf, 200, TEI)))
        self.download = self.enterContext(patch.object(FastPDFDownloader, '_download_pdf_attempt_async',
                                                      new_callable=AsyncMock, return_value=None))

    def corpus(self, count, downloaded):
        ids = sorted(str(i) for i in range(count))
        self.s2.search_snippets = Mock(return_value={'data': [{
            'paper': {'corpusId': cid, 'title': 'Paper ' + cid},
            'snippet': {'text': 'Saved search evidence ' + cid, 'annotations': {'refMentions': []}},
        } for cid in ids]})
        self.s2.get_paper_bulk = Mock(return_value=[{'paperId': 's2-' + cid, 'title': 'Paper ' + cid,
            'abstract': 'Saved abstract ' + cid, 'openAccessPdf': {'url': 'https://example.org/' + cid + '.pdf'}} for cid in ids])
        for cid in ids[:downloaded]:
            write_pdf(self.pdfs / (cid + '.pdf'))
        return ids

    def run_retrieval(self):
        retrieve(self.args, self.s2, self.progress, self.progress.state['metrics'])
        return json.loads((self.root / 'snippet_papers.json').read_text())

    def test_31_of_37_continue_and_keep_unavailable_evidence(self):
        ids = self.corpus(37, 31)
        papers = self.run_retrieval()
        self.assertEqual(set(papers), set(ids))
        self.assertEqual(self.process.call_count, 31)
        self.assertEqual(self.download.await_count, 6)
        for cid in ids[31:]:
            paper = papers[cid]
            self.assertEqual(paper['paper_id'], 's2-' + cid)
            self.assertEqual(paper['download_status'], 'full_text_unavailable')
            self.assertEqual(paper['evidence_status'], 'metadata_abstract_only')
            self.assertEqual(paper['metadata']['abstract'], 'Saved abstract ' + cid)
            self.assertIn('Saved search evidence', paper['paper'])
            self.assertEqual(paper['attempted_urls'], ['https://example.org/' + cid + '.pdf'])
            self.assertIn('exhausted', paper['failure_reason'])
        coverage = json.loads((self.root / 'snippet_coverage.json').read_text())
        self.assertEqual((coverage['full_text_downloaded'], coverage['parsed_successfully'],
                          coverage['full_text_unavailable'], coverage['parse_failed']), (31, 31, 6, 0))
        self.assertEqual(coverage['status'], 'completed')
        cp = StageCheckpoint('ScholarEval.soundness.snippet_search', [
            '--queries_file', self.args.queries_file, '--methods_file', self.args.methods_file,
            '--output_file', self.args.output_file])
        cp.complete()
        self.assertTrue(cp.valid())
        papers.pop(ids[-1])
        atomic_json(self.root / 'snippet_papers.json', papers)
        self.assertFalse(cp.validate())

    def test_resume_reuses_pdfs_parses_and_exhausted_downloads(self):
        self.corpus(3, 2)
        first = self.run_retrieval()
        hashes = {p.name: p.read_bytes() for p in self.pdfs.glob('*.pdf')}
        self.download.reset_mock()
        self.process.reset_mock()
        second = self.run_retrieval()
        self.download.assert_not_awaited()
        self.process.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(hashes, {p.name: p.read_bytes() for p in self.pdfs.glob('*.pdf')})

    def test_individual_parse_failure_continues_remaining_documents(self):
        self.corpus(3, 3)
        self.process.side_effect = lambda service, pdf: (pdf, 422, None) if Path(pdf).stem == '1' else (pdf, 200, TEI)
        papers = self.run_retrieval()
        self.assertEqual(self.process.call_count, 3)
        self.assertEqual(papers['1']['parse_status'], 'parse_failed')
        self.assertEqual(papers['1']['evidence_status'], 'metadata_abstract_only')
        self.assertIn('422', papers['1']['failure_reason'])
        self.assertEqual(sum(p['evidence_status'] == 'full_text_available' for p in papers.values()), 2)
        self.process.reset_mock()
        self.run_retrieval()
        self.process.assert_not_called()

    def test_malformed_xml_is_individual_parse_failure(self):
        self.corpus(2, 2)
        self.process.side_effect = lambda service, pdf: (pdf, 200, '<TEI>') if Path(pdf).stem == '1' else (pdf, 200, TEI)
        papers = self.run_retrieval()
        self.assertEqual(papers['1']['parse_status'], 'parse_failed')
        self.assertEqual(papers['0']['evidence_status'], 'full_text_available')

    def test_all_network_search_failure_still_fails(self):
        self.corpus(2, 2)
        self.s2.search_snippets.side_effect = RetrievalNetworkError('network unavailable')
        with self.assertRaises(RetrievalNetworkError):
            self.run_retrieval()
        self.process.assert_not_called()
        self.download.assert_not_awaited()
        self.assertFalse((self.root / 'snippet_papers.json').exists())

    def test_metadata_batch_failure_still_fails(self):
        self.corpus(2, 2)
        self.s2.get_paper_bulk.side_effect = RetrievalNetworkError('network unavailable')
        with self.assertRaises(RetrievalNetworkError):
            self.run_retrieval()
        self.process.assert_not_called()

    def test_truncated_metadata_batch_fails(self):
        self.corpus(2, 2)
        self.s2.get_paper_bulk.return_value = []
        with self.assertRaises(RetrievalResponseError):
            self.run_retrieval()

    def test_no_pdf_url_preserves_candidate(self):
        self.corpus(1, 0)
        self.s2.get_paper_bulk.return_value[0]['openAccessPdf'] = None
        papers = self.run_retrieval()
        self.assertEqual(papers['0']['download_status'], 'full_text_unavailable')
        self.assertEqual(papers['0']['attempted_urls'], [])
        self.download.assert_not_awaited()
        self.service.ensure_ready.assert_not_called()

    def test_no_full_text_permitted_by_default_optional_minimum_is_one(self):
        self.corpus(1, 0)
        self.assertEqual(self.run_retrieval()['0']['evidence_status'], 'metadata_abstract_only')
        with patch.dict(os.environ, {'SCHOLAREVAL_FULL_TEXT_POLICY': 'require-full-text'}):
            from ScholarEval.utils.retrieval_http import RetrievalError
            with self.assertRaisesRegex(RetrievalError, 'at least one'):
                self.run_retrieval()

    def test_systemic_grobid_failure_still_fails_and_saves_retryable_state(self):
        self.corpus(2, 2)
        self.process.side_effect = lambda service, pdf: (pdf, 500, None)
        with self.assertRaises(GrobidStartupError):
            self.run_retrieval()
        state = json.loads((self.root / 'snippet_parse_progress.json').read_text())
        self.assertTrue(all(p['status'] == 'parse_retryable' for p in state['papers'].values()))
        self.assertEqual(len(json.loads((self.root / 'snippet_papers.json').read_text())), 2)

    def test_service_dies_after_partial_success_preserves_parsed_document(self):
        self.corpus(2, 2)
        self.service.healthy.return_value = False
        self.process.side_effect = lambda service, pdf: (pdf, 500, None) if Path(pdf).stem == '1' else (pdf, 200, TEI)
        with self.assertRaises(GrobidStartupError):
            self.run_retrieval()
        papers = json.loads((self.root / 'snippet_papers.json').read_text())
        self.assertEqual(papers['0']['evidence_status'], 'full_text_available')
        self.assertEqual(papers['1']['parse_status'], 'parse_retryable')

    def test_optional_full_text_policy_needs_one_parsed_document_not_all(self):
        self.corpus(3, 1)
        with patch.dict(os.environ, {'SCHOLAREVAL_FULL_TEXT_POLICY': 'require-full-text'}):
            papers = self.run_retrieval()
        self.assertEqual(sum(p['evidence_status'] == 'full_text_available' for p in papers.values()), 1)

    def test_busy_grobid_retry_is_bounded(self):
        self.corpus(1, 1)
        self.process.side_effect = lambda service, pdf: (pdf, 503, None)
        with patch('ScholarEval.utils.grobid.time.sleep'), self.assertRaises(GrobidStartupError):
            self.run_retrieval()
        self.assertEqual(self.process.call_count, 2)

    def test_tls_fallback_verifies_certificates_and_records_failure(self):
        downloader = FastPDFDownloader(pdf_dir=str(self.pdfs))
        response = Mock(status_code=403, url='https://example.org/p.pdf')
        with patch('ScholarEval.utils.pdf_utils.requests.get', return_value=response) as request:
            asyncio.run(downloader._requests_fallback(response.url, str(self.pdfs / 'p.pdf'), 'p'))
        self.assertIs(request.call_args.kwargs['verify'], True)
        record = {'attempted_urls': [], 'failures': []}
        downloader.active_downloads['p'] = record
        with patch('ScholarEval.utils.pdf_utils.requests.get', side_effect=requests.exceptions.SSLError('bad certificate')):
            asyncio.run(downloader._requests_fallback(response.url, str(self.pdfs / 'p.pdf'), 'p'))
        self.assertTrue(any(f['reason'] == 'SSLError' for f in record['failures']))


class GrobidTransportTests(unittest.TestCase):
    def test_client_does_not_misclassify_network_timeout_as_bad_document(self):
        def init(client, **kwargs):
            client.config = {'timeout': 1, 'grobid_server': 'http://fixture:8070'}
            client.logger = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / 'one.pdf'
            write_pdf(pdf)
            service = Mock(metrics={}, healthy=Mock(return_value=True), diagnostics=Mock(return_value='fixture'))
            results = {}
            with patch('ScholarEval.utils.grobid.GrobidService', return_value=service), \
                 patch('grobid_client.grobid_client.GrobidClient.__init__', init), \
                 patch('grobid_client.grobid_client.GrobidClient.post', side_effect=requests.ReadTimeout('fixture')), \
                 self.assertRaises(GrobidStartupError):
                parse_pdf_corpus([pdf], Path(tmp), lambda cid, record: results.update({cid: record}))
            self.assertEqual(results['one']['http_status'], 408)
            self.assertEqual(results['one']['status'], 'parse_retryable')
            self.assertIn('ReadTimeout', results['one']['failure_detail'])


if __name__ == '__main__':
    unittest.main()
