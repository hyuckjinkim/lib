import os
import sys

# # 예시
# install_packages = ['pandas','numpy','seaborn']
# PackageInstall(
#     install_packages=install_packages,
#     upgrade_packages=None,
#     verbose=True,
# )
def PackageInstall(install_packages=None,upgrade_packages=None,verbose=True):

    #---------------------------------------------------------------------------------------------------------------------------#
    # 1. Import Modules
    #---------------------------------------------------------------------------------------------------------------------------#
    if verbose:
        print('')
        print('#### 1. Install Modules ####\n')
    if install_packages is not None:
        for i,package in enumerate(install_packages):
            if verbose:
                print('({}/{}) {}'.format(i+1,len(install_packages),package))
            try:
                exec(f'import {package}')
            except:
                os.system(f'pip install -q {package}')

    #---------------------------------------------------------------------------------------------------------------------------#
    # 2. Upgrade Modules
    #---------------------------------------------------------------------------------------------------------------------------#
    if verbose:
        print('')
        print('#### 2. Upgrade Modules ####\n')
    if upgrade_packages is not None:
        for i,package in enumerate(upgrade_packages):
            if verbose:
                print('({}/{}) {}'.format(i+1,len(upgrade_packages),package))
            os.system(f'pip uninstall -q {package} -y')
            os.system(f'pip install -q --upgrade {package}')
    
    # ## 1.3. numpy downgrade for tensorflow
    # os.system('pip uninstall numpy -y')
    # os.system('pip install -U numpy==1.19.5')

    ## Process Done
    if verbose:
        print('')
        print('##################')
        print('## Process Done ##')
        print('##################')