import argparse
import os
import urllib.parse

import requests
from tqdm import tqdm


class YandexDiskDownloader:
    def __init__(self, link: str, download_location: str) -> None:
        self.link = link
        self.download_location = download_location

    def download(self) -> None:
        url = f'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={self.link}'
        response = requests.get(url, timeout=30)
        download_url = response.json()['href']
        file_name = urllib.parse.unquote(download_url.split('filename=')[1].split('&')[0])
        save_path = os.path.join(self.download_location, file_name)

        with requests.get(download_url, stream=True, timeout=30) as download_response:
            total_size = int(download_response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
            with open(save_path, 'wb') as file_handle:
                for chunk in download_response.iter_content(chunk_size=1024):
                    if chunk:
                        file_handle.write(chunk)
                        progress_bar.update(len(chunk))
            progress_bar.close()

        print('Download complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yandex Disk Downloader')
    parser.add_argument('-l', '--link', type=str, help='Link for Yandex Disk URL', required=True)
    parser.add_argument(
        '-d',
        '--download_location',
        type=str,
        help='Download location in PC',
        default='.',
        required=False,
    )
    args = parser.parse_args()

    downloader = YandexDiskDownloader(args.link, args.download_location)
    downloader.download()
