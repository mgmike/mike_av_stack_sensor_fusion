from setuptools import setup

package_name = 'mike_av_stack_sensor_fusion'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mike',
    maintainer_email='mgmike1023@aol.com',
    description='Michael\'s autonomous vehicle stack built on top of carla ros bridge',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_fusion = mike_av_stack_sensor_fusion.sensor_fusion:main',
        ],
    },
)
