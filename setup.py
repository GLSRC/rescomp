"""
"""

from distutils.core import setup

setup(
    name='rescomp',
    version='2019-11-22',
    description='',  # TODO
    long_description='',  # TODO
    keywords='',  # TODO
    author='Sebastian Baur',  # TODO
    author_email='sebastian.baur@dlr.de',
    maintainer='Sebastian Baur',
    maintainer_email='sebastian.baur@dlr.de',
    url='http://www.dlr.de/',
    download_url='',  # TODO
    # package_dir = {'': 'src'},
    packages=[
        'rcdlr'],
    # scripts=[
    #     'src/scripts/staub_dummy.py',
    #     'src/scripts/staub_create_image.py',
    #     'src/scripts/staub_find_particles.py',
    #     'src/scripts/staub_display_garching_format.py',
    #     'src/scripts/staub_enhance_images.py',
    #     'src/scripts/staub_find_template.py'],
    license='',  # TODO
    classifiers=[
        'Development Status :: 3 - Alpha',
        # 'Environment :: Console',
        # 'Environment :: X11 Applications',
        'Intended Audience :: Science/Research',
        # 'License :: OSI Approved :: GNU General Public License (GPL)', # TODO
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: BSD :: FreeBSD',
        'Operating System :: POSIX :: BSD :: OpenBSD',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        # 'Topic :: Software Development :: Libraries :: Python Modules' # TODO
    ],
    # cat $(find | grep "py$") | egrep -i "^[ \t]*import .*$" | egrep -i --only-matching "import .*$" | sort -u
    requires=[  # TODO
        'matplotlib',
        'numpy',
        'scipy'
        'networkx'
        'time'
        'datetime'
        'pickle'
        'matplotlib'

        ],
    provides=['rescomp']
)
