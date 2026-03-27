from setuptools import setup
import os
from glob import glob

package_name = 'rl_agent'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nwc232',
    maintainer_email='nwc836@gmail.com',
    description='RL driving agent for F1Tenth',
    license='MIT',
    entry_points={
        'console_scripts': [
            'rl_node = rl_agent.rl_node:main',
            'train = rl_agent.train:main',
        ],
    },
)