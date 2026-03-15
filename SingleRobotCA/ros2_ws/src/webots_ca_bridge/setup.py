from setuptools import setup
import os
from glob import glob

package_name = 'webots_ca_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vision-CA',
    maintainer_email='user@example.com',
    description='ROS2 bridge for Webots vision-based CA simulation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bridge_node = webots_ca_bridge.bridge_node:main',
        ],
    },
)
