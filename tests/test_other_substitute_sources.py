import unittest

from scripts.probe_other_experience_substitutes import candidates
from test_substitute_sources import ORIGINAL, doc, package


class OtherSubstituteSourceTests(unittest.TestCase):
    def test_only_requirements_in_supported_diagnostic_formats(self):
        result = candidates(package([doc(2, format="docx"), doc(3, format="zip"),
            doc(4, format="rar"), doc(5), doc(6, format="zip", title="Documentos de Evaluación")]), ORIGINAL)
        self.assertEqual([r["format"] for r in result], ["docx", "zip", "rar"])
        self.assertTrue(all(not r["ranking_eligible"] for r in result))

    def test_original_url_unsafe_host_and_other_process_are_not_substitutes(self):
        self.assertEqual(candidates(package([doc(format="docx", url=ORIGINAL["url"]),
            doc(3, format="zip", url="https://example.com/file.zip")]), ORIGINAL), [])
        with self.assertRaises(ValueError):
            candidates(package([doc(format="docx")], ocid="ocds-other-1"), ORIGINAL)
