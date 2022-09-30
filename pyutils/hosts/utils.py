import os


def get_hostname():
    # Get the hostname
    if str(os.uname().nodename).__contains__('.'):
        hostname = str(os.uname().nodename).split('.')[0]
    else:
        hostname = str(os.uname().nodename)
    return hostname
