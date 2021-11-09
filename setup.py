from distutils.core import setup
from glob import glob

setup(name='machine-learning-utils',
      version=str(2021.0),
      description="""Machine Learning Utils Package for supporting the Introduction to Machine Learning Course at Imperial College London""",
      author="Feifan Fan",
      author_email="frankfanff53@gmail.com",
      packages=["ml_utils"],
      scripts=glob('scripts/*'))
