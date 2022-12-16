@rem install_ce_conda-packages.cmd
@rem =============================
@rem Windows Command Script to install the recommended packages and versions
@rem for a Python installation at CE.

@rem Update conda
call conda update conda

@rem Set Python version
call conda install python=3.6.9

@rem Update pip
call conda update pip

@rem conda packages
call conda install -c defaults -n base^
  alabaster=0.7.12^
  asn1crypto=1.2.0^
  astroid=2.3.3^
  attrs=19.3.0^
  babel=2.7.0^
  backcall=0.1.0^
  blas=1.0^
  bleach=3.1.0^
  ca-certificates=2020.1.1^
  certifi=2020.4.5.1^
  cffi=1.13.2^
  chardet=3.0.4^
  click=7.0^
  cloudpickle=1.2.2^
  colorama=0.4.1^
  coverage=4.5.4^
  cryptography=2.8^
  cycler=0.10.0^
  decorator=4.4.1^
  defusedxml=0.6.0^
  docutils=0.15.2^
  entrypoints=0.3^
  et_xmlfile=1.0.1^
  freetype=2.9.1^
  future=0.18.2^
  icc_rt=2019.0.0^
  icu=58.2^
  idna=2.8^
  imageio=2.6.1^
  imagesize=1.1.0^
  importlib_metadata=0.23^
  intel-openmp=2019.4^
  ipykernel=5.1.3^
  ipython=7.9.0^
  ipython_genutils=0.2.0^
  ipywidgets=7.5.1^
  isort=4.3.21^
  itsdangerous=1.1.0^
  jdcal=1.4.1^
  jedi=0.15.1^
  jinja2=2.10.3^
  jpeg=9b^
  jsonschema=3.1.1^
  jupyter=1.0.0^
  jupyter_client=5.3.4^
  jupyter_console=6.0.0^
  jupyter_core=4.6.1^
  jupyterlab=1.2.6^
  jupyterlab_server=1.1.4^
  keyring=18.0.0^
  kiwisolver=1.1.0^
  lazy-object-proxy=1.4.3^
  libpng=1.6.37^
  libsodium=1.0.16^
  libtiff=4.1.0^
  m2w64-gcc-libgfortran=5.3.0^
  m2w64-gcc-libs=5.3.0^
  m2w64-gcc-libs-core=5.3.0^
  m2w64-gmp=6.1.0^
  m2w64-libwinpthread-git=5.0.0.4634.697f757^
  markupsafe=1.1.1^
  matplotlib=3.1.1^
  mccabe=0.6.1^
  mistune=0.8.4^
  mkl=2019.4^
  mkl-service=2.3.0^
  mkl_fft=1.0.15^
  mkl_random=1.1.0^
  more-itertools=7.2.0^
  msys2-conda-epoch=20160418^
  nbconvert=5.6.1^
  nbformat=4.4.0^
  nose=1.3.7^
  notebook=6.0.2^
  numpy=1.17.3^
  numpy-base=1.17.3^
  numpydoc=0.9.1^
  olefile=0.46^
  openpyxl=3.0.1^
  openssl=1.1.1g^
  packaging=19.2^
  pandas=1.0.3^
  pandoc=2.2.3.2^
  pandocfilters=1.4.2^
  parso=0.5.1^
  patsy=0.5.1^
  pickleshare=0.7.5^
  pillow=6.2.1^
  plotly=4.2.1^
  prometheus_client=0.7.1^
  prompt_toolkit=2.0.10^
  psutil=5.6.5^
  pycodestyle=2.5.0^
  pycparser=2.19^
  pyflakes=2.1.1^
  pygments=2.4.2^
  pylint=2.4.4^
  pyopenssl=19.0.0^
  pyparsing=2.4.5^
  pyqt=5.9.2^
  pyrsistent=0.15.5^
  pysocks=1.7.1^
  python=3.6.9^
  python-dateutil=2.8.1^
  pytz=2019.3^
  pywin32=223^
  pywinpty=0.5.5^
  pyyaml=5.1.2^
  pyzmq=18.1.0^
  qt=5.9.7^
  qtawesome=0.6.0^
  qtconsole=4.5.5^
  qtpy=1.9.0^
  requests=2.22.0^
  retrying=1.3.3^
  rope=0.14.0^
  scipy=1.3.1^
  send2trash=1.5.0^
  setuptools=41.6.0^
  sip=4.19.8^
  six=1.13.0^
  snowballstemmer=2.0.0^
  sphinx=2.2.1^
  sphinxcontrib-applehelp=1.0.1^
  sphinxcontrib-devhelp=1.0.1^
  sphinxcontrib-htmlhelp=1.0.2^
  sphinxcontrib-jsmath=1.0.1^
  sphinxcontrib-qthelp=1.0.2^
  sphinxcontrib-serializinghtml=1.1.3^
  spyder=3.3.6^
  spyder-kernels=0.5.2^
  sqlite=3.30.1^
  statsmodels=0.10.1^
  terminado=0.8.2^
  testpath=0.4.4^
  tornado=6.0.3^
  traitlets=4.3.3^
  typed-ast=1.4.0^
  urllib3=1.24.2^
  vc=14.1^
  vs2015_runtime=14.16.27012^
  wcwidth=0.1.7^
  webencodings=0.5.1^
  werkzeug=0.16.0^
  wheel=0.33.6^
  widgetsnbextension=3.5.1^
  win_inet_pton=1.1.0^
  wincertstore=0.2^
  winpty=0.4.3^
  wrapt=1.11.2^
  xlrd=1.2.0^
  xlsxwriter=1.2.6^
  xlwt=1.3.0^
  xz=5.2.4^
  yaml=0.1.7^
  zeromq=4.3.1^
  zipp=0.6.0^
  zlib=1.2.11^
  zstd=1.3.7

call conda install -c conda-forge -n base^
  dash=1.4.1^
  dash-core-components=1.3.1^
  dash-html-components=1.0.1^
  dash-renderer=1.1.2^
  dash-table=4.4.1^
  tk=8.6.9^
  flask=1.1.1^
  flask-compress=1.4.0

