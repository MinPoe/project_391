import os
from glob import glob
from setuptools import setup

package_name = 'milestone3'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hamzah',
    maintainer_email='hamzah.chaudhry@proton.me',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'safety_node = milestone3.safety_node:main',
            'wall_node = milestone3.wall_node:main',
            'gap_node = milestone3.gap_node:main',
            'cam_node = milestone3.cam_node:main',
            'lap_counter = milestone3.lap_counter:main',
            'safety_node_old = milestone3.safety_node_old:main',
            'gap_node_old = milestone3.gap_node_old:main'
        ],
    },
)
