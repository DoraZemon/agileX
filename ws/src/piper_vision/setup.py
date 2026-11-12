from glob import glob

from setuptools import find_packages, setup

package_name = 'piper_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sanlanx',
    maintainer_email='sanlanx@users.noreply.github.com',
    description='Camera config, hand-eye calibration and L1 / L2 detection for PiPER',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['piper_vision = piper_vision.cli:main'],
    },
)
