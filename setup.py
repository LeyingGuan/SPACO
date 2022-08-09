from setuptools import setup, find_packages

VERSION = '1.1.0'
DESCRIPTION = 'Smooth and probablistic PARAFAC decomposition with covariates'
LONG_DESCRIPTION = 'Smooth and probablistic PARAFAC decomposition with covariates'


setup(name='spaco',
      version=VERSION,
      description=DESCRIPTION,
      author='Leying Guan',
      author_email='leying.guan@yale.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=['setuptools==59.5.0','mxnet','torch==1.11.0','tensorflow==2.3.1','numpy==1.18.0','pandas==1.2.1',
                         'sklearn', 'rpy2==3.4.2', 'tensorly==0.6.0'],
      keywords=['python', 'spaco'],
      classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers'],
      zip_safe=False)
