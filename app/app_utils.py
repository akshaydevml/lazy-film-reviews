import os
import urllib

from tqdm import tqdm


def download_weights(url, out_path):
    response = getattr(urllib, 'request', urllib).urlopen(url)
    filename = url.split('/')[-1]
    with tqdm.wrapattr(open(os.path.join(out_path, filename), "wb"),
                       "write",
                       miniters=1,
                       desc=filename,
                       position=0,
                       leave=True,
                       total=getattr(response, 'length', None)) as fout:
        for chunk in response:
            fout.write(chunk)
