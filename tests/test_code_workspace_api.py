from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from skills import code_workspace as cw


class WorkspaceAPITests(unittest.TestCase):
    def test_downloads_stay_in_session_and_are_attachments(self):
        from webui import app as module
        with tempfile.TemporaryDirectory() as root, patch.object(module, '_ROOT', Path(root)):
            file = cw.write_file(root, 'S1', 'notes.md', '<script>not HTML</script>')
            client = module.app.test_client()
            listing = client.get('/api/code-workspace/S1')
            self.assertEqual(listing.status_code, 200)
            self.assertEqual(listing.json['files'][0]['path'], 'files/notes.md')
            response = client.get(file['download_url'])
            self.assertEqual(response.status_code, 200)
            self.assertIn('attachment', response.headers['Content-Disposition'])
            self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
            self.assertIn('application/octet-stream', response.content_type)
            response.close()
            self.assertEqual(client.get('/api/code-workspace/S1/download?path=../../configs/llm.json').status_code, 400)
            (cw.folder(root, 'S1')/'files/link.txt').symlink_to('/etc/passwd')
            self.assertEqual(client.get('/api/code-workspace/S1/download?path=files/link.txt').status_code, 400)
            self.assertEqual(client.get('/api/code-workspace/S2/download?path=files/notes.md').status_code, 404)
