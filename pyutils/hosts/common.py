import platform

from pyutils.hosts.utils import get_hostname

HOSTNAME = get_hostname()
PLATFORM_ARCH = platform.processor()
