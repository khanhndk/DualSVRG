from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import sys
import shutil
import hashlib
import tarfile
# from six.moves.urllib.request import urlopen
# from six.moves.urllib.error import URLError, HTTPError
# import six.moves.urllib as urllib
# from urllib.request import urlopen
from urllib.error import URLError, HTTPError
# import urllib
import requests

# Under Python 2, 'urlretrieve' relies on FancyURLopener from legacy
# urllib module, known to have issues with proxy management
# if sys.version_info[0] == 2:


def urlretrieve(url, filename, reporthook=None, data=None, total_size=None):
    def chunk_read(response, chunk_size=8192, reporthook=None, total_size=None):
        if total_size is None:
            if 'Content-length' in response.headers.keys():
                total_size = int(response.headers['content-length'])
            else:
                total_size = 0
        with open(filename, 'wb') as fd:
            count = 0
            for chunk in response.iter_content(chunk_size):
                count += 1
                if not chunk:
                    reporthook(count, total_size, total_size)
                    break
                if reporthook:
                    reporthook(count, chunk_size, total_size)
                fd.write(chunk)

    s = requests.Session()
    res1 = s.get(url, stream=True)

    chunk_read(res1, reporthook=reporthook, total_size=total_size)
    # print(os.stat(filename))
    if os.stat(filename).st_size < 10000:
        print('Get new link ...')
        with open(filename, 'r') as my_file:
            data = my_file.read()
        result = re.search('(confirm=.*?)(&)', data)
        confirm = result.group(1)
        url += '&' + confirm

        # Use the cookie is subsequent requests
        res2 = s.get(url, stream=True)
        chunk_read(res2, reporthook=reporthook, total_size=total_size)


def get_file(fname, origin, untar=False,
             md5_hash=None, cache_subdir='', total_size=None):
    """Downloads a file from a URL if it not already in the cache.

    Passing the MD5 hash will verify the file after download
    as well as if it is already present in the cache.

    # Arguments
        fname: name of the file
        origin: original URL of the file
        untar: boolean, whether the file should be decompressed
        md5_hash: MD5 hash of the file for verification
        cache_subdir: directory being used as the cache

    # Returns
        Path to the downloaded file
    """
    from ..configs import data_dir
    from .generic_utils import Progbar

    datadir = os.path.join(data_dir(), cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # file found; verify integrity if a hash was provided
        if md5_hash is not None:
            if not validate_file(fpath, md5_hash):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)
        global progbar
        progbar = None

        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size, show_steps=1)
            else:
                progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress, total_size=total_size)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        progbar = None
        print()

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
        return untar_fpath

    return fpath


def data_info(dataset):
    w8a = dict()
    w8a['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1-8_DR6XryBMR0As71cQjSC-p1Fk-Qf-Q&export=download'
    w8a['md5_hash'] = '74e119dafbc8a4f1fedf2a7745cb4e9f'
    w8a['size'] = 3500000

    codrna = dict()
    codrna['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1Rpd_b1ImEb3B2XFgATOpS7wacMYuLAQk&export=download'
    codrna['md5_hash'] = '28e33991b6d3ffa3cd6c5ca48377f8da'
    codrna['size'] = 13000000
    covtype = dict()
    covtype['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1TRt4v6r4Zqy8i3XOym0OJNV0SS3nqeSN&export=download'
    covtype['md5_hash'] = 'bbfd64fe4421f6ac6394cbecbf774566'
    covtype['size'] = 17000000

    airline = dict()
    airline['origin'] = 'https://drive.google.com/a/hcmup.edu.vn/uc?id=1TTuMEUSeH1Qbmk8446flpgx7WMuJMPM-&export=download'
    airline['md5_hash'] = '18c9c88d47f4e6b9f9e778b483b06ad0'
    airline['size'] = 142000000

    info = dict()
    info['w8a'] = w8a
    info['cod-rna'] = codrna
    info['covtype'] = covtype
    info['airline.2008'] = airline
    return info[dataset]


def validate_file(fpath, md5_hash):
    '''Validates a file against a MD5 hash

    # Arguments
        fpath: path to the file being validated
        md5_hash: the MD5 hash being validated against

    # Returns
        Whether the file is valid
    '''
    hasher = hashlib.md5()
    with open(fpath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    if str(hasher.hexdigest()) == str(md5_hash):
        return True
    else:
        return False


def scp_file(filepath):
    pass


if __name__ == '__main__':
    scp_file(sys.argv[1])
