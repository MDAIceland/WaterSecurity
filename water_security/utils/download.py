import urllib
from utils.notebook import isnotebook

if isnotebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """
    Progress Bar for downloading purposes
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url_while_in_notebook(url, output_path):
    import threading
    from IPython.display import display

    progress = DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    )

    thread = threading.Thread(
        target=download_url_while_not_in_notebook, args=(url, output_path, progress)
    )
    display(progress)
    thread.start()
    return thread


def download_url_while_not_in_notebook(url: str, output_path: str, widget=None) -> None:
    """
    Download a file from a url and save it to the provided output path
    """
    if widget is None:
        widget = DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        )
    with widget as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if isnotebook:
    download_url = download_url_while_in_notebook
else:
    download_url = download_url_while_not_in_notebook
