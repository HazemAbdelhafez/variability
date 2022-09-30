from os.path import join as jp

from pyutils.common.paths import DATA_DIR
from pyutils.hosts.common import HOSTNAME

ONE_VS_MANY_DIR = jp(DATA_DIR, 'one-vs-many')
MANY_DATA_DIR = jp(ONE_VS_MANY_DIR, HOSTNAME, 'many')
ONE_DATA_DIR = jp(ONE_VS_MANY_DIR, HOSTNAME, 'one')
COMMON_DATA_DIR = jp(ONE_VS_MANY_DIR, HOSTNAME, 'common')
