import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info
    import numpy

    # needs to be called during build otherwise show_version may fail sometimes
    get_info('blas_opt', 0)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('mltools', parent_package, top_path)

    # submodules which do not have their own setup.py
    config.add_subpackage('featureEngineering')
    config.add_subpackage('evaluateModels')
    config.add_subpackage('stackingEnsemble')
    config.add_subpackage('timeSeriesTools')
    config.add_subpackage('textMining')
    config.add_subpackage('plottingTool')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())