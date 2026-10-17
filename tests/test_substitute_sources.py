import copy
import unittest
from tesis_recommender.substitute_sources import select_substitute

ORIGINAL={'opportunity_id':'ocds-test-1','sample_number':25,'document_id':'original',
          'url':'https://prod1.seace.gob.pe/SeaceWeb-PRO/SdescargarArchivoAlfresco?fileCode=00000000-0000-0000-0000-000000000001',
          'initial_version_published_at':'2025-01-02T00:00:00-05:00'}


def doc(n=2,**changes):
    return dict({'id':str(n),'documentType':'biddingDocuments','title':'Bases Integradas','format':'pdf',
                 'url':ORIGINAL['url'][:-1]+str(n),'datePublished':'2025-01-10T00:00:00-05:00'},**changes)


def package(docs,ocid='ocds-test-1'):
    return {'records':[{'ocid':ocid,'compiledRelease':{'ocid':ocid,'tender':{'documents':docs}}}]}


class SubstituteSourceTests(unittest.TestCase):
    def test_later_integrated_bases_are_separate_development_not_initial_ranking(self):
        r=select_substitute(package([doc()]),ORIGINAL)['selected']
        self.assertEqual(r['temporal_metadata_status'],'declared_after_or_modified_after_initial')
        self.assertFalse(r['eligible_for_initial_time_ranking']);self.assertFalse(r['historical_content_proven'])
        self.assertEqual(r['replaces_original_document'],ORIGINAL)

    def test_prior_metadata_preferred_without_claiming_historical_bytes(self):
        r=select_substitute(package([doc(),doc(3,title='Términos de referencia',datePublished='2025-01-01T00:00:00-05:00')]),ORIGINAL)['selected']
        self.assertEqual(r['document_id'],'3');self.assertFalse(r['ranking_eligible'])

    def test_modification_after_initial_is_not_initial_time_metadata(self):
        r=select_substitute(package([doc(datePublished='2025-01-01T00:00:00-05:00',dateModified='2025-02-01T00:00:00-05:00')]),ORIGINAL)['selected']
        self.assertEqual(r['temporal_metadata_status'],'declared_after_or_modified_after_initial')

    def test_original_url_not_retried_even_with_new_identity(self):
        self.assertIsNone(select_substitute(package([doc(url=ORIGINAL['url'])]),ORIGINAL)['selected'])

    def test_excludes_awards_proposals_evaluation_archives_and_other_host(self):
        for changes in [{'documentType':'awardNotice'},{'title':'Documentos de Presentación de Propuestas'},
                        {'title':'Documentos de Evaluación'}, {'format':'zip'},{'url':'https://example.com/document.pdf'}]:
            self.assertIsNone(select_substitute(package([doc(**changes)]),ORIGINAL)['selected'])

    def test_unknown_and_timezone_free_dates_abstain(self):
        for value in [None,'bad','2025-01-01']:
            self.assertIsNone(select_substitute(package([doc(datePublished=value)]),ORIGINAL)['selected'])

    def test_missing_identity_and_bad_modification_date_abstain(self):
        for changes in [{'id':None},{'dateModified':'bad'}]:
            self.assertIsNone(select_substitute(package([doc(**changes)]),ORIGINAL)['selected'])

    def test_other_process_or_duplicate_record_rejected(self):
        with self.assertRaises(ValueError):select_substitute(package([doc()],ocid='ocds-other-1'),ORIGINAL)
        p=package([doc()]);p['records']*=2
        with self.assertRaises(ValueError):select_substitute(p,ORIGINAL)

    def test_duplicate_document_metadata_is_not_extra_candidate(self):
        r=select_substitute(package([doc(),doc()]),ORIGINAL)
        self.assertEqual(len(r['eligible_candidates']),1)
        self.assertEqual(r['excluded'][0]['reason'],'duplicate_document_metadata')

    def test_same_id_changed_url_does_not_prove_same_bytes(self):
        r=select_substitute(package([doc(id='original')]),ORIGINAL)['selected']
        self.assertTrue(r['same_document_identity_declared']);self.assertFalse(r['historical_content_proven'])

    def test_deterministic_metadata_selection_and_no_input_mutation(self):
        p=package([doc(3),doc(2)]);before=copy.deepcopy(p)
        self.assertEqual(select_substitute(p,ORIGINAL)['selected']['document_id'],'2')
        self.assertEqual(p,before)
