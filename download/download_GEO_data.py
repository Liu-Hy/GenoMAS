import functools
import json
import logging
import os
import re
import socket
import sys
import time
import traceback
from datetime import datetime
from ftplib import FTP
from typing import Optional, List, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"geo_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FTPConfig:
    CONNECT_TIMEOUT = 600  # 120 for max 100 datasets  # Initial connection timeout
    DATA_TIMEOUT = 600  # Data transfer timeout
    MAX_RETRIES = 2
    INITIAL_DELAY = 1
    MAX_DELAY = 4
    BLOCK_SIZE = 32768
    SIZE_LIMIT_MB = 100


def checkpoint_read(path: str, show: bool = True) -> Tuple[int, int]:
    if not os.path.exists(path):
        if show:
            logger.info("No checkpoint found, starting from beginning")
        with open(path, 'w') as f:
            f.write("0;0")
        return 0, 0

    try:
        with open(path, 'r') as f:
            checkpoint = f.read()
            last_checkpoint_i, last_checkpoint_j = map(int, checkpoint.strip().split(";"))
        if show:
            logger.info(f"Resuming from checkpoint: i={last_checkpoint_i}, j={last_checkpoint_j}")
        return last_checkpoint_i, last_checkpoint_j
    except Exception as e:
        logger.error(f"Error reading checkpoint, starting from beginning: {str(e)}\n{traceback.format_exc()}")
        return 0, 0


def clean_filename(filename: str) -> str:
    pattern = r'[\\/:\*\?"<>\|\r\n\t]+'  # Matches \ / : * ? " < > | \r \n \t
    filename = re.sub(pattern, '-', filename)
    filename = '_'.join(filename.split())
    filename = ''.join(filename.split("'"))
    return filename


# FTP linking information

def retry_on_ftp_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_attempts = FTPConfig.MAX_RETRIES
        attempt = 0
        while attempt < max_attempts:
            try:
                return func(*args, **kwargs)
            except (socket.timeout, EOFError, ConnectionError) as e:
                attempt += 1
                if attempt == max_attempts:
                    raise
                logger.error(f"Connection error: {str(e)}. Retrying... ({attempt}/{max_attempts})")
                time.sleep(1)  # Fixed 1-second delay for simplicity
        return None

    return wrapper


@retry_on_ftp_error
def ftp_connect(host, timeout=FTPConfig.CONNECT_TIMEOUT):  # Reduced initial connection timeout
    """Establish FTP connection"""
    ftp = FTP(host, timeout=timeout)
    ftp.login()
    ftp.set_pasv(True)  # Added passive mode for better compatibility
    return ftp


@retry_on_ftp_error
def download_files(ftp, remote_file_paths, local_file_paths):
    """Download files"""
    for remote_file_path, local_file_path in zip(remote_file_paths, local_file_paths):
        try:
            with open(local_file_path, 'wb') as file:
                ftp.retrbinary('RETR ' + remote_file_path, file.write, blocksize=FTPConfig.BLOCK_SIZE)
            logger.info(f"Successfully downloaded: {os.path.basename(local_file_path)}")
        except Exception as e:
            logger.error(f"Failed to download {remote_file_path}: {str(e)}\n{traceback.format_exc()}")
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            raise


def get_GEO_series_access(search_term: str, batch_size: int = 100,
                          max_results: Optional[int] = None) -> Tuple[List[str], List[str], List[int]]:
    """Search and retrieve GEO datasets for a given trait.
    
    Filters for human studies (2010-2025) with 30-10000 samples.
    Includes expression profiling and genome variation datasets.
    
    Returns:
        Tuple of (dataset_accessions, ftp_links, sample_sizes)
    """
    try:
        Entrez.email = os.getenv("ENTREZ_EMAIL")
        Entrez.api_key = os.getenv("ENTREZ_API_KEY")

        handle = None
        DATASET_TYPES = [
            "expression profiling by array",
            "expression profiling by high throughput sequencing",
            "genome variation profiling by high throughput sequencing",
            "genome variation profiling by snp array",
            "snp genotyping by snp array",
            "third party reanalysis"
        ]

        dataset_filter = "(" + " OR ".join(
            [f"\"{dataset_type}\"[DataSet Type]" for dataset_type in DATASET_TYPES]) + ")"

        query = (
            f"(30:10000[Number of Samples])"
            f" AND (\"Homo sapiens\"[Organism])"
            f" AND (2010:2025[Publication Date])"
            f" AND ({search_term})"
            f" AND {dataset_filter}"
        )

        logger.info(f"Search query: {query}")

        try:
            # First pass: retmax=0 to only get the count of matching records
            handle = Entrez.esearch(db="gds", term=query, retmax=0, timeout=30)
            record_first_pass = Entrez.read(handle)
            handle.close()

            total_count = int(record_first_pass["Count"])
            logger.info(f"Total results for query: {total_count}")

            # If user wants to cap results
            if max_results is not None and total_count > max_results:
                total_count = max_results
                logger.info(f"Limiting results to user-specified maximum: {max_results}")

            all_ids = []
            retstart = 0
            total_batches = (total_count - 1) // batch_size + 1

            # Iteratively fetch IDs in batches
            while retstart < total_count:
                try:
                    current_batch_size = min(batch_size, total_count - retstart)
                    current_batch = retstart // batch_size + 1

                    logger.info(f"Fetching batch {current_batch} of {total_batches} (size: {current_batch_size})")

                    handle = Entrez.esearch(
                        db="gds",
                        term=query,
                        retmax=current_batch_size,
                        retstart=retstart,
                        timeout=30
                    )
                    record = Entrez.read(handle)
                    handle.close()

                    if "IdList" in record:
                        batch_ids = record["IdList"]
                        all_ids.extend(batch_ids)
                        logger.info(f"Successfully retrieved {len(batch_ids)} IDs in batch {current_batch}")
                    else:
                        logger.warning(f"No IdList found in record batch at retstart={retstart}")

                    retstart += current_batch_size

                    # Add rate limiting between batches
                    if retstart < total_count:
                        time.sleep(0.5)  # 500ms delay between batches

                except Exception as e:
                    logger.error(f"Error in batch {current_batch}: {str(e)}\n{traceback.format_exc()}")
                    break

            logger.info(f"Found {len(all_ids)} datasets for: {search_term}")

        except Exception as e:
            logger.error(f"Error searching GEO database: {str(e)}\n{traceback.format_exc()}")
            return [], [], []
        finally:
            if handle:
                handle.close()

        dataset_accessions = []
        Series_FTP_Links = []
        sample_sizes = []

        for dataset_id in all_ids:
            try:
                dataset_handle = Entrez.esummary(db="gds", id=dataset_id)
                dataset_summary = Entrez.read(dataset_handle)
                if not dataset_summary:
                    logger.warning(f"Empty summary for dataset {dataset_id}")
                    continue

                sample_size = int(dataset_summary[0]['n_samples'])
                dataset_accession = dataset_summary[0]['Accession']
                ftp_link = dataset_summary[0].get('FTPLink', '')

                if not ftp_link:
                    logger.warning(f"No FTP link for dataset {dataset_id}")
                    continue

                dataset_accessions.append(dataset_accession)
                Series_FTP_Links.append(ftp_link)
                sample_sizes.append(sample_size)

            except Exception as e:
                logger.error(f"Error processing dataset {dataset_id}: {str(e)}\n{traceback.format_exc()}")
                continue
            finally:
                dataset_handle.close()

        logger.info(f"Successfully processed {len(dataset_accessions)} datasets for: {search_term}")
        return dataset_accessions, Series_FTP_Links, sample_sizes

    except Exception as e:
        logger.error(f"Unexpected error in GEO access: {str(e)}\n{traceback.format_exc()}")
        return [], [], []


def download_trait_data(dataset_inf, key_word, base_dir, checkpoint_path, max_download_per_trait=10):
    """Download GEO datasets for a given trait.
    
    For each dataset, downloads:
    - Matrix files (expression/variation data)
    - Family files (metadata)
    Limited to max_download_per_trait datasets, with size checks.
    """
    ftp_host = 'ftp.ncbi.nlm.nih.gov'
    ftp = None
    downloaded_sample_sizes = []  # Track sample sizes of downloaded datasets
    try:
        ftp = ftp_connect(ftp_host)
        dataset_accessions, Series_FTP_Links, sample_sizes = dataset_inf
        local_dir = os.path.join(base_dir, clean_filename(key_word))
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        Series_num = len(Series_FTP_Links)
        _, checkpoint_j = checkpoint_read(checkpoint_path)
        for j, Series_FTP_Link in enumerate(Series_FTP_Links):
            if j < checkpoint_j:
                continue

            try:
                parsed_url = urlparse(Series_FTP_Link)
                ftp_path = parsed_url.path

                try:
                    ftp.cwd(ftp_path)
                except:
                    try:
                        ftp.quit()
                    except:
                        pass
                    ftp = ftp_connect(ftp_host)
                    ftp.cwd(ftp_path)

                file_list = ftp.nlst()
                logger.debug(f"File list: {file_list}")

                matrix_flag = False
                family_flag = False
                for file_name in file_list:
                    if "matrix" in file_name:
                        matrix_flag = True
                    if "soft" in file_name:
                        family_flag = True
                if not matrix_flag or not family_flag:
                    logger.debug(f"Skipping series {j}: missing matrix or family files")
                    continue

                # Process matrix files
                ftp.cwd(f'{ftp_path}matrix/')
                matrix_file_list = ftp.nlst()
                logger.debug(f"Matrix files: {matrix_file_list}")
                matrix_file_urls = []
                matrix_file_names = []
                for filename1 in matrix_file_list:
                    if 'matrix' in filename1 and 'xml' not in filename1:
                        ftp.sendcmd("TYPE I")
                        matrix_file_size = ftp.size(filename1) / (1024 * 1024)
                        if matrix_file_size > 0.1 and matrix_file_size < FTPConfig.SIZE_LIMIT_MB:
                            logger.info(f"Matrix file {filename1} is available with size {matrix_file_size:.2f} MB")
                            matrix_file_url = f'{ftp_path}matrix/{filename1}'
                            matrix_file_urls.append(matrix_file_url)
                            matrix_file_names.append(filename1)

                # Process family files
                ftp.cwd(f'{ftp_path}soft/')
                family_file_list = ftp.nlst()
                logger.debug(f"Family files: {family_file_list}")
                family_file_urls = []
                family_file_names = []
                for filename2 in family_file_list:
                    if 'family' in filename2 and 'xml' not in filename2:
                        ftp.sendcmd("TYPE I")
                        family_file_size = ftp.size(filename2) / (1024 * 1024)
                        if family_file_size < FTPConfig.SIZE_LIMIT_MB:
                            logger.info(f"Family file {filename2} is available with size {family_file_size:.2f} MB")
                            family_file_url = f'{ftp_path}soft/{filename2}'
                            family_file_urls.append(family_file_url)
                            family_file_names.append(filename2)

                if len(family_file_urls) > 0 and len(matrix_file_urls) > 0:
                    local_dir_series = os.path.join(local_dir, dataset_accessions[j])
                    if not os.path.exists(local_dir_series):
                        os.makedirs(local_dir_series)

                    local_matrix_filenames = [os.path.join(local_dir_series, mfn) for mfn in matrix_file_names]
                    download_files(ftp, remote_file_paths=matrix_file_urls, local_file_paths=local_matrix_filenames)

                    local_family_filenames = [os.path.join(local_dir_series, ffn) for ffn in family_file_names]
                    download_files(ftp, remote_file_paths=family_file_urls, local_file_paths=local_family_filenames)

                    logger.info(f"Downloaded: {matrix_file_names} and {family_file_names} for trait {key_word}")
                    downloaded_sample_sizes.append(sample_sizes[j])
                    if len(downloaded_sample_sizes) >= max_download_per_trait:
                        break
                    # Save the index of the downloaded dataset to checkpoint file
                    checkpoint_i, _ = checkpoint_read(checkpoint_path, show=False)
                    with open(checkpoint_path, 'w') as f:
                        f.write(f"{checkpoint_i};{(j + 1)}")
                else:
                    logger.info(
                        f"No suitable gene expression data in series {j} ({Series_num} series for trait {key_word} in total)")

            except Exception as e:
                logger.error(f"Error processing series {j}: {str(e)}\n{traceback.format_exc()}")
                continue
    finally:
        if ftp:
            try:
                ftp.quit()
            except:
                pass
        return downloaded_sample_sizes


def read_keywords() -> List[str]:
    """Read trait keywords from metadata 'task_info'.
    
    Returns a list of traits to search in GEO database.
    """
    file_path = "../metadata/task_info.json"
    try:
        with open(file_path, "r") as f:
            task_info = json.load(f)
            return sorted(list(task_info.keys()))
    except Exception as e:
        logger.error(f"Error reading keywords file: {str(e)}\n{traceback.format_exc()}")
        return []


if __name__ == '__main__':
    """Search and download GEO datasets for each trait.
    
    Reads traits from the metadata, searches GEO database,
    and downloads relevant datasets with checkpointing.
    """
    key_word = read_keywords()
    logger.info(f"Found {len(key_word)} keywords to process")
    logger.info(f"Keywords: {key_word}")

    checkpoint_path = "./GEO_data_download_Checkpoint.txt"

    last_checkpoint_i, last_checkpoint_j = checkpoint_read(checkpoint_path)
    try:
        sample_sizes = []
        total_traits = len(key_word)

        with tqdm(total=total_traits, desc="Traits Progress") as traits_pbar:
            if last_checkpoint_i > 0:
                traits_pbar.update(last_checkpoint_i)

            for idx, keyword in enumerate(key_word):
                if idx < last_checkpoint_i:
                    continue
                logger.info(f"\nProcessing trait ({idx + 1}/{total_traits}): {keyword}")

                try:
                    data_info = get_GEO_series_access(search_term=keyword, max_results=200)

                    if not data_info[1]:  # Series_FTP_Links
                        logger.warning(f"No valid datasets found for trait: {keyword}")
                        traits_pbar.update(1)
                        continue

                    trait_downloaded_sizes = download_trait_data(dataset_inf=data_info, key_word=keyword,
                                                                 base_dir='/media/techt/DATA/GEO_test',
                                                                 checkpoint_path=checkpoint_path,
                                                                 max_download_per_trait=10)
                    sample_sizes.extend(trait_downloaded_sizes)

                    with open(checkpoint_path, 'w') as g:
                        g.write(f"{idx + 1};0")

                    traits_pbar.update(1)

                except Exception as e:
                    logger.error(f"Error processing keyword {keyword}: {str(e)}\n{traceback.format_exc()}")
                    raise

        if sample_sizes:
            logger.info("\nFinal Statistics:")
            logger.info(f"Mean samples: {np.mean(sample_sizes):.2f}")
            logger.info(f"Std dev: {np.std(sample_sizes):.2f}")
            logger.info(f"Total datasets: {len(sample_sizes)}")
        else:
            logger.warning("No sample sizes were collected during execution.")

    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}\n{traceback.format_exc()}")
