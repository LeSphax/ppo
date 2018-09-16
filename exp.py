from experiment_runner import push_experiment

from docopt import docopt

_USAGE = '''
Usage:
    exp (<label>) (<env_name>) [<arguments>...]
'''
options = docopt(_USAGE)

label = str(options['<label>'])
env_name = str(options['<env_name>'])
arguments = options['<arguments>']

command = 'python3 my_ppo.py {0} {1} {2}'.format(label, env_name, '' if len(arguments) == 0 else '--' + ' --'.join([str(x) for x in arguments]))
print(command)
push_experiment('/home/sphax/Desktop/ML/Experiments/ppo', '/home/sphax/Desktop/ML/Experiments/experiment-runner', label, command)
