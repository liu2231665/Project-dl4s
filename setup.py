from setuptools import setup, find_packages

setup(name='dl4stochastic',
      version='0.0.0',
      url='https://github.com/liu2231665/dl4stochastic.git',
      license='MIT',
      author='Yingru Liu',
      author_email='liu2231665@hotmail.com',
      description='',
      packages=find_packages(exclude=['tests']),
      long_description=open('README.md').read(),
      zip_safe=False,
      setup_requires=['nose>=1.0', 'tensorflow'],
      install_requires=['nose>=1.0', 'tensorflow'],
      test_suite='nose.collector'
      )