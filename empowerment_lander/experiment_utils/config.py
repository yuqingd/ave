import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')

DOCKER_MOUNT_DIR = '/root/code/data'

DOCKER_IMAGE = '/empowerment_lander:latest'

S3_BUCKET_NAME = 'empowerment.lander'
