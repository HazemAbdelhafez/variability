import os
import subprocess as sp
import sys
from os.path import join as jp

from pyutils.common.paths import PROJECT_DIR, DATA_DIR
from pyutils.common.utils import GlobalLogger

logger = GlobalLogger().get_logger()

BACKUP_DATA_DIR = jp(PROJECT_DIR, 'backups')

# Archive cmd
generated_data_dir = os.path.basename(DATA_DIR)


def get_part_number(prefix):
    all_files = os.listdir(BACKUP_DATA_DIR)
    archives = list()
    for f in all_files:
        if not f.__contains__('.tar.gz') or not f.startswith(prefix):
            continue
        part_number = int(f.split('.')[0].split('_')[-1])
        archives.append(part_number)

    # Get last part number
    if len(archives) != 0:
        return max(archives) + 1
    else:
        return 0


def get_paths(prefix, part_number):
    archive_file_path = jp(BACKUP_DATA_DIR, f'{prefix}_{part_number}.tar.gz')
    meta_file_path = jp(BACKUP_DATA_DIR, f'{prefix}_backup.meta')
    return archive_file_path, meta_file_path


def run_archive_cmd(archive_file_path, meta_file_path):
    arch_cmd = ['tar', '-czvg', meta_file_path, '-f', archive_file_path, DATA_DIR]
    logger.info(f"Running backup command: {str(arch_cmd)}")
    completed_process = sp.run(arch_cmd, capture_output=True, text=True)
    if completed_process.returncode != 0:
        logger.error(completed_process.stderr)
        logger.error(completed_process.stdout)
    else:
        logger.info(completed_process.stdout)
        logger.info(completed_process.stderr)


def run_restore_cmd(archive_file_path, unpacking_dir):
    cmd = ['tar', '-xvf', archive_file_path, '-C', unpacking_dir]
    logger.info(f"Running restore command: {str(cmd)}")
    completed_process = sp.run(cmd, capture_output=True, text=True)
    if completed_process.returncode != 0:
        logger.error(completed_process.stderr)
        logger.error(completed_process.stdout)
    else:
        logger.info(completed_process.stdout)
        logger.info(completed_process.stderr)


def restore(prefix, num_of_parts):
    for i in range(num_of_parts):
        archive_file_path, _ = get_paths(prefix, i)
        logger.info(f"Restoring: {archive_file_path}")
        restore_dir = jp(PROJECT_DIR, 'restores')
        if not os.path.exists(restore_dir):
            os.makedirs(restore_dir)
        run_restore_cmd(archive_file_path, unpacking_dir=restore_dir)


def entry(mode='backup'):
    prefix = generated_data_dir
    logger.info(f"Prefix:            {prefix}")
    part_number = get_part_number(prefix=prefix)
    logger.info(f"Number of parts:       {part_number}")
    if mode == 'backup':

        archive_file_path, meta_file_path = get_paths(prefix=prefix, part_number=part_number)
        logger.info(f"Archive file path: {archive_file_path}")
        logger.info(f"Meta file path:    {meta_file_path}")
        if os.path.exists(archive_file_path):
            logger.error(f"Archive file path: {archive_file_path} already exists. Aborting.")
        run_archive_cmd(archive_file_path, meta_file_path)
    else:
        restore(prefix, part_number)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        entry(mode='backup')
    else:
        entry(mode=str(sys.argv[1]))
