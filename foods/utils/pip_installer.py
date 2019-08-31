import subprocess

def subp_pip_install(modules):
    import subprocess
    if isinstance(modules, str): modules = modules.split(' ')
    return subprocess.run(['pip', 'install', '--source-directory=/tmp'] + modules,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
