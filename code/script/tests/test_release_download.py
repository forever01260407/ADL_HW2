"""Protect model integrity and archive extraction at the download boundary."""
import hashlib
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

spec=importlib.util.spec_from_file_location('download_release',Path(__file__).parents[1]/'download_release.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseDownloadTests(unittest.TestCase):
    def archive(self,root,members):
        path=root/'model.zip'
        with zipfile.ZipFile(path,'w') as z:
            for name,data in members:
                z.writestr(name,data)
        return path,module.digest(path)

    def test_hash_verified_install_and_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p,h=self.archive(root,[('weights.bin',b'weights'),('1_Pooling/config.json',b'{}')])
            files={'weights.bin':hashlib.sha256(b'weights').hexdigest(),
                '1_Pooling/config.json':hashlib.sha256(b'{}').hexdigest()}
            module.install_archive(p,root/'model',files,h)
            module.install_archive(p,root/'model',files,h)
            self.assertTrue(module.matches(root/'model',files))

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p,h=self.archive(root,[('../escape',b'bad')])
            with self.assertRaises(ValueError):
                module.install_archive(p,root/'model',{},h)
            self.assertFalse((root/'escape').exists())

    def test_tampered_weights_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p,h=self.archive(root,[('weights.bin',b'wrong')])
            with self.assertRaises(ValueError):
                module.install_archive(p,root/'model',{'weights.bin':hashlib.sha256(b'right').hexdigest()},h)
            self.assertFalse((root/'model').exists())

    def test_existing_different_model_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p,h=self.archive(root,[('weights.bin',b'right')])
            dest=root/'model';dest.mkdir();(dest/'weights.bin').write_bytes(b'old')
            with self.assertRaises(FileExistsError):
                module.install_archive(p,dest,{'weights.bin':hashlib.sha256(b'right').hexdigest()},h)
            self.assertEqual((dest/'weights.bin').read_bytes(),b'old')


if __name__=='__main__':
    unittest.main()
