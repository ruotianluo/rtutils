import setuptools

#with open("README.md", "r") as fh:
#   long_description = fh.read()

setuptools.setup(
    name="rtutils", # Replace with your own username
    version="0.2",
    author="Ruotian Luo",
    author_email="rluo@ttic.edu",
    entry_points={
        'console_scripts': [
            'rtdrive=rtutils.drive:main',
            'gd=rtutils_cli.gdrive_wrapper:main'
        ]
    },
#    description="A small example package",
#    long_description=long_description,
#    long_description_content_type="text/markdown",
#   url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
#   classifiers=[
#       "Programming Language :: Python :: 3",
#       "License :: OSI Approved :: MIT License",
#       "Operating System :: OS Independent",
#   ],
#   python_requires='>=3.6',
)
