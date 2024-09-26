from setuptools import setup, find_packages
from glob import glob

package_name = 'cuboid_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/sample_data', glob('dataset/mesh/*.obj')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ksy',
    maintainer_email='sangyoon@g.skku.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cuboid_detector = script.cuboid_detector:main',
        ],
    },
)
