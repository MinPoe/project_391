import os
from glob import glob
from setuptools import setup

package_name = 'project'

setup(
    name=package_name,
    version='0.0.0',
    package_dir={package_name: 'nodes', 'project.bc': 'bc', 'project.sac': 'sac'},
    packages=[package_name, 'project.bc', 'project.sac'],
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
            'safety_node = project.safety_node:main',
            'bc_inference_node = project.bc_inference_node:main',
            'sac_inference_node = project.sac_inference_node:main'
        ],
    },
)
