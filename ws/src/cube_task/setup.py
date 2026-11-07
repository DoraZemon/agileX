from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'cube_task'

TOOLS = [
    'run_task', 'check_setup', 'capture', 'view', 'calib_points', 'calib_verify',
    'teach_poses', 'teach_classes', 'color_calib', 'sim_bench',
]


def _files(pattern):
    return [f for f in glob(pattern) if os.path.isfile(f)]


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', _files('config/*.yaml')),
        ('share/' + package_name + '/config/examples', _files('config/examples/*.yaml')),
        ('share/' + package_name + '/scripts', _files('scripts/*.sh')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zph',
    maintainer_email='1319500935@qq.com',
    description='PiPER vision pick-and-place for the Songling Cup L1 / L2 / L3 tasks',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            '%s = cube_task.tools.%s:main' % (t, t) for t in TOOLS
        ],
    },
)
