"""Model store which provides pretrained models."""
from __future__ import print_function

__all__ = ['get_model_file', 'purge']

import os
import zipfile
import logging
import portalocker

from ..utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    ('4ff279e78cdb0b8bbc9363181df6f094ad46dc36', 'mobilenet_yolo.param'),
    ('1528cf08b9823fc01aaebfc932ec8c8d4a3b1613', 'mobilenet_yolo.bin'),
    ('3f5b78b0c982f8bdf3a2c3a27e6136d4d2680e96', 'mobilenetv2_yolov3.param'),
    ('0705b0f8fe5a77718561b9b7d6ed4f33fcd3d455', 'mobilenetv2_yolov3.bin'),
    ('3723ce3e312db6a102cff1a5c39dae80e1de658e', 'mobilenet_ssd_voc_ncnn.param'),
    ('8e2d2139550dcbee1ce5e200b7697b25aab29656', 'mobilenet_ssd_voc_ncnn.bin'),
    ('52c669821dc32ef5b7ab30749fa71a3bc27786b8', 'squeezenet_ssd_voc.param'),
    ('347e31d1cbe469259fa8305860a7c24a95039202', 'squeezenet_ssd_voc.bin'),
    ('52dab628ecac8137e61ce3aea1a912f9c5a0a638', 'mobilenetv2_ssdlite_voc.param'),
    ('9fea06f74f7c60d753cf703ea992f92e50a986d4', 'mobilenetv2_ssdlite_voc.bin'),
    ('f36661eff1eda1e36185e7f2f28fc722ad8b66bb', 'mobilenetv3_ssdlite_voc.param'),
    ('908f63ca9bff0061a499512664b9c533a0b7f485', 'mobilenetv3_ssdlite_voc.bin'),
    ('a63d779a1f789af976bc4e2eae86fdd9b0bb6c2c', 'squeezenet_v1.1.param'),
    ('262f0e33e37aeac69021b5a3556664be65fc0aeb', 'squeezenet_v1.1.bin'),
    ('3ba57cccd1d4a583f6eb76eae25a2dbda7ce7f74', 'ZF_faster_rcnn_final.param'),
    ('1095fbb5f846a1f311b40941add5fef691acaf8d', 'ZF_faster_rcnn_final.bin'),
    ('3586ec3d663b1cc8ec8c662768caa9c7fbcf4fdc', 'pelee.param'),
    ('2442ad483dc546940271591b86db0d9c8b1c7118', 'pelee.bin'),
    ('6cfeda08d5494a1274199089fda77c421be1ecac', 'mnet.25-opt.param'),
    ('3ff9a51dc81cdf506a87543dbf752071ffc50b8d', 'mnet.25-opt.bin'),
    ('50acebff393c91468a73a7b7c604ef231429d068', 'rfcn_end2end.param'),
    ('9a68cd937959b4dda9c5bf9c99181cb0e40f266b', 'rfcn_end2end.bin'),
    ('5a76b44b869a925d64abefcadf296c0f36886085', 'shufflenet_v2_x0.5.param'),
    ('85998dfe1fb2caeeadc6267927434bb5aa0878f3', 'shufflenet_v2_x0.5.bin'),
    ('7c8f8d72c60aab6802985423686b36c61be2f68c', 'pose.param'),
    ('7f691540972715298c611a3e595b20c59c2147ce', 'pose.bin'),
]}

apache_repo_url = 'https://github.com/caishanli/pyncnn-assets/raw/master/models/'
_url_format = '{repo_url}{file_name}'


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def get_model_file(name, tag=None, root=os.path.join('~', '.ncnn', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.ncnn/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    if 'NCNN_HOME' in os.environ:
        root = os.path.join(os.environ['NCNN_HOME'], 'models')

    use_tag = isinstance(tag, str)
    if use_tag:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=tag)
    else:
        file_name = '{name}'.format(name=name)

    root = os.path.expanduser(root)
    params_path = os.path.join(root, file_name)
    lockfile = os.path.join(root, file_name + '.lock')
    if use_tag:
        sha1_hash = tag
    else:
        sha1_hash = _model_sha1[name]

    if not os.path.exists(root):
        os.makedirs(root)

    with portalocker.Lock(lockfile, timeout=int(os.environ.get('NCNN_MODEL_LOCK_TIMEOUT', 300))):
        if os.path.exists(params_path):
            if check_sha1(params_path, sha1_hash):
                return params_path
            else:
                logging.warning("Hash mismatch in the content of model file '%s' detected. "
                                "Downloading again.", params_path)
        else:
            logging.info('Model file not found. Downloading.')

        zip_file_path = os.path.join(root, file_name)
        repo_url = os.environ.get('NCNN_REPO', apache_repo_url)
        if repo_url[-1] != '/':
            repo_url = repo_url + '/'
        download(_url_format.format(repo_url=repo_url, file_name=file_name),
                 path=zip_file_path,
                 overwrite=True)
        if zip_file_path.endswith(".zip"):
            with zipfile.ZipFile(zip_file_path) as zf:
                zf.extractall(root)
            os.remove(zip_file_path)
        # Make sure we write the model file on networked filesystems
        try:
            os.sync()
        except AttributeError:
            pass
        if check_sha1(params_path, sha1_hash):
            return params_path
        else:
            raise ValueError('Downloaded file has different hash. Please try again.')


def purge(root=os.path.join('~', '.ncnn', 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.ncnn/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".params"):
            os.remove(os.path.join(root, f))
