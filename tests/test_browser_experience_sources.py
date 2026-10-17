import copy
import unittest
from scripts.register_browser_experience_sources import validate_observation


class BrowserSourceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.selected = {"opportunity_id": "ocid", "title": "Bases Integradas",
            "url": "https://prod1.seace.gob.pe/SeaceWeb-PRO/SdescargarArchivoAlfresco?fileCode=file-uuid",
            "date_published_declared": "2025-05-14T12:06:49-05:00"}
        self.tender = {"id": "1110599", "title": "CP-SM-6-2025-DRTC/CS-1"}
        self.observation = {"opportunity_id": "ocid", "convocation_id": "1110599",
            "nomenclature": self.tender["title"], "file_code": "file-uuid",
            "document_title_displayed": "Bases Integradas", "document_date_displayed": "14/05/2025 12:06",
            "ficha_url": "https://prod2.seace.gob.pe/seacebus-uiwd-pub/fichaSeleccion/fichaSeleccion.xhtml?idHistorial=e850ce8c-01ec-446a-a8f0-52aa095bf423&ptoRetorno=listaHistorialContrataciones",
            "size_kb_displayed": 11534, "status": "not_downloaded_displayed_size_over_limit"}
        self.limits = {"max_bytes_per_document": 10485760}

    def check(self, **updates):
        observation = copy.deepcopy(self.observation)
        observation.update(updates)
        return validate_observation(observation, self.selected, self.tender, self.limits)

    def test_valid_size_exclusion(self):
        self.check()

    def test_valid_received_download(self):
        self.check(size_kb_displayed=2015, status="browser_download_received", download_path="/tmp/file.pdf")

    def test_wrong_file_identity(self):
        with self.assertRaises(ValueError):
            self.check(file_code="other")

    def test_wrong_process_identity(self):
        with self.assertRaises(ValueError):
            self.check(convocation_id="999")

    def test_wrong_nomenclature(self):
        with self.assertRaises(ValueError):
            self.check(nomenclature="another procedure")

    def test_wrong_date(self):
        with self.assertRaises(ValueError):
            self.check(document_date_displayed="14/05/2025 12:07")

    def test_no_size_limit_bypass(self):
        with self.assertRaises(ValueError):
            self.check(status="browser_download_received", download_path="/tmp/file.pdf")

    def test_no_session_parameters_saved(self):
        with self.assertRaises(ValueError):
            self.check(ficha_url=self.observation["ficha_url"] + "&session=anything")

    def test_no_other_destination(self):
        with self.assertRaises(ValueError):
            self.check(ficha_url=self.observation["ficha_url"].replace("prod2.seace.gob.pe", "other.example"))

    def test_no_path_without_download(self):
        with self.assertRaises(ValueError):
            self.check(download_path="/tmp/file.pdf")


if __name__ == "__main__":
    unittest.main()
