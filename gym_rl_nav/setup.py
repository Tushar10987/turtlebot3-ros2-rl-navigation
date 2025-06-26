import os
from setuptools import setup

package_name = 'gym_rl_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/sim_env.launch.py',
        ]),
        ('share/' + package_name + '/worlds', [
            'worlds/nav_world.world',
        ]),
    ],
    install_requires=[
        'setuptools',
        'gym',
        'rclpy',
        'numpy',
    ],
    zip_safe=True,
    maintainer='tushar',
    maintainer_email='tushar@example.com',
    description='Custom gym + ROS 2 environment for reinforcement learning-based navigation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'test_env = gym_rl_nav.test.test_env:main',
        ],
    },
)

