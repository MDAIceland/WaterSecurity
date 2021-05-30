import threading
import urllib
from typing import Tuple

from utils.notebook import isnotebook

if isnotebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from collections import deque


class DownloadProgressBar(tqdm):
    """
    Progress Bar for downloading purposes
    """

    def __init__(self, *args, download_queue=None, **kwargs):
        self.download_queue = download_queue
        self.cnt = 0
        super().__init__(self, *args, **kwargs)

    def update_to(self, b=1, bsize=1, tsize=None):
        global DOWNLOAD_PROGRESS_
        if tsize is not None:
            self.total = tsize

        self.update(b * bsize - self.n)
        self.cnt += bsize
        if self.download_queue is not None:
            self.download_queue.append(
                f"{round(100*self.cnt/1024**2)/100}MB/{round(100*self.total/1024**2)/100}MB"
            )


def download_url_while_in_notebook(url, output_path) -> Tuple[threading.Thread, deque]:
    """
    Returns download thread and list that contains download progress string
    """
    from IPython.display import display

    queue = deque(maxlen=10)
    progress = DownloadProgressBar(
        unit="B",
        unit_scale=True,
        miniters=1,
        desc=url.split("/")[-1],
        download_queue=queue,
    )
    thread = threading.Thread(
        target=download_url_while_not_in_notebook,
        args=(url, output_path, progress),
    )
    thread.start()
    display(progress)
    return thread, queue


def download_url_while_not_in_notebook(
    url: str, output_path: str, widget=None, queue=None
) -> None:
    """
    Download a file from a url and save it to the provided output path
    """
    if widget is None:
        widget = DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=url.split("/")[-1],
            download_queue=queue,
        )
    with widget as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if isnotebook:
    download_url = download_url_while_in_notebook
else:
    download_url = download_url_while_not_in_notebook
