# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Model store which provides pretrained models."""
from __future__ import print_function

__all__ = ["get_model_file", "purge"]

import os
import zipfile
import logging
import portalocker

from ..utils import download, check_sha1

_model_sha1 = {
    name: checksum
    for checksum, name in [
        ("4ff279e78cdb0b8bbc9363181df6f094ad46dc36", "mobilenet_yolo.param"),
        ("1528cf08b9823fc01aaebfc932ec8c8d4a3b1613", "mobilenet_yolo.bin"),
        ("3f5b78b0c982f8bdf3a2c3a27e6136d4d2680e96", "mobilenetv2_yolov3.param"),
        ("0705b0f8fe5a77718561b9b7d6ed4f33fcd3d455", "mobilenetv2_yolov3.bin"),
        ("de59186323ebad5650631e12a6cc66b526ec7df4", "yolov4-tiny-opt.param"),
        ("1765c3b251c041dd6ac59d2ec3ddf7b983fe9ee9", "yolov4-tiny-opt.bin"),
        ("e92d3a3a8ac5e6a6c08c433aa2252b0680124328", "yolov4-opt.param"),
        ("69d128b42b70fb790e9d3ccabcf1b6e8cc2859fe", "yolov4-opt.bin"),
        ("6fa8ccc8cabc0f5633ab3c6ffa268e6042b8888f", "yolov5s.param"),
        ("0cbab3664deb090480ea748c1305f6fe850b9ac4", "yolov5s.bin"),
        ("e65bae7052d9e9b9d45e1214a8d1b5fe6f64e8af", "yolact.param"),
        ("9bda99f50b1c14c98c5c6bbc08d4f782eed66548", "yolact.bin"),
        ("3723ce3e312db6a102cff1a5c39dae80e1de658e", "mobilenet_ssd_voc_ncnn.param"),
        ("8e2d2139550dcbee1ce5e200b7697b25aab29656", "mobilenet_ssd_voc_ncnn.bin"),
        ("52c669821dc32ef5b7ab30749fa71a3bc27786b8", "squeezenet_ssd_voc.param"),
        ("347e31d1cbe469259fa8305860a7c24a95039202", "squeezenet_ssd_voc.bin"),
        ("52dab628ecac8137e61ce3aea1a912f9c5a0a638", "mobilenetv2_ssdlite_voc.param"),
        ("9fea06f74f7c60d753cf703ea992f92e50a986d4", "mobilenetv2_ssdlite_voc.bin"),
        ("f36661eff1eda1e36185e7f2f28fc722ad8b66bb", "mobilenetv3_ssdlite_voc.param"),
        ("908f63ca9bff0061a499512664b9c533a0b7f485", "mobilenetv3_ssdlite_voc.bin"),
        ("a63d779a1f789af976bc4e2eae86fdd9b0bb6c2c", "squeezenet_v1.1.param"),
        ("262f0e33e37aeac69021b5a3556664be65fc0aeb", "squeezenet_v1.1.bin"),
        ("3ba57cccd1d4a583f6eb76eae25a2dbda7ce7f74", "ZF_faster_rcnn_final.param"),
        ("1095fbb5f846a1f311b40941add5fef691acaf8d", "ZF_faster_rcnn_final.bin"),
        ("3586ec3d663b1cc8ec8c662768caa9c7fbcf4fdc", "pelee.param"),
        ("2442ad483dc546940271591b86db0d9c8b1c7118", "pelee.bin"),
        ("6cfeda08d5494a1274199089fda77c421be1ecac", "mnet.25-opt.param"),
        ("3ff9a51dc81cdf506a87543dbf752071ffc50b8d", "mnet.25-opt.bin"),
        ("50acebff393c91468a73a7b7c604ef231429d068", "rfcn_end2end.param"),
        ("9a68cd937959b4dda9c5bf9c99181cb0e40f266b", "rfcn_end2end.bin"),
        ("d6b289cda068e9a9d8a171fb909352a05a39a494", "shufflenet_v2_x0.5.param"),
        ("2ccd631d04a1b7e05483cd8a8def76bca7d330a8", "shufflenet_v2_x0.5.bin"),
        ("7c8f8d72c60aab6802985423686b36c61be2f68c", "pose.param"),
        ("7f691540972715298c611a3e595b20c59c2147ce", "pose.bin"),
        ("979d09942881cf1207a93cbfa9853005a434469b", "nanodet_m.param"),
        ("51d868905361e4ba9c45bd12e8a5608e7aadd1bd", "nanodet_m.bin"),
    ]
}


_split_model_bins = {
    "ZF_faster_rcnn_final.bin": 3,
    "rfcn_end2end.bin": 2,
    "yolov4-opt.bin": 7,
}


github_repo_url = "https://github.com/nihui/ncnn-assets/raw/master/models/"
_url_format = "{repo_url}{file_name}"


def merge_file(root, files_in, file_out, remove=True):
    with open(file_out, "wb") as fd_out:
        for file_in in files_in:
            file = os.path.join(root, file_in)
            with open(file, "rb") as fd_in:
                fd_out.write(fd_in.read())
            if remove == True:
                os.remove(file)


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError(
            "Pretrained model for {name} is not available.".format(name=name)
        )
    return _model_sha1[name][:8]


def get_model_file(name, tag=None, root=os.path.join("~", ".ncnn", "models")):
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
    if "NCNN_HOME" in os.environ:
        root = os.path.join(os.environ["NCNN_HOME"], "models")

    use_tag = isinstance(tag, str)
    if use_tag:
        file_name = "{name}-{short_hash}".format(name=name, short_hash=tag)
    else:
        file_name = "{name}".format(name=name)

    root = os.path.expanduser(root)
    params_path = os.path.join(root, file_name)
    lockfile = os.path.join(root, file_name + ".lock")
    if use_tag:
        sha1_hash = tag
    else:
        sha1_hash = _model_sha1[name]

    if not os.path.exists(root):
        os.makedirs(root)

    with portalocker.Lock(
        lockfile, timeout=int(os.environ.get("NCNN_MODEL_LOCK_TIMEOUT", 300))
    ):
        if os.path.exists(params_path):
            if check_sha1(params_path, sha1_hash):
                return params_path
            else:
                logging.warning(
                    "Hash mismatch in the content of model file '%s' detected. "
                    "Downloading again.",
                    params_path,
                )
        else:
            logging.info("Model file not found. Downloading.")

        zip_file_path = os.path.join(root, file_name)
        if file_name in _split_model_bins:
            file_name_parts = [
                "%s.part%02d" % (file_name, i + 1)
                for i in range(_split_model_bins[file_name])
            ]
            for file_name_part in file_name_parts:
                file_path = os.path.join(root, file_name_part)
                repo_url = os.environ.get("NCNN_REPO", github_repo_url)
                if repo_url[-1] != "/":
                    repo_url = repo_url + "/"
                download(
                    _url_format.format(repo_url=repo_url, file_name=file_name_part),
                    path=file_path,
                    overwrite=True,
                )

            merge_file(root, file_name_parts, zip_file_path)
        else:
            repo_url = os.environ.get("NCNN_REPO", github_repo_url)
            if repo_url[-1] != "/":
                repo_url = repo_url + "/"
            download(
                _url_format.format(repo_url=repo_url, file_name=file_name),
                path=zip_file_path,
                overwrite=True,
            )
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
            raise ValueError("Downloaded file has different hash. Please try again.")


def purge(root=os.path.join("~", ".ncnn", "models")):
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
