run_my_solution = False

import os
import copy
import signal
import os
import numpy as np

if run_my_solution:
    from A3mysolution import *
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')

else:
    
    import subprocess, glob, pathlib

    assignmentNumber = '3'

    nb_name = '*A{}*.ipynb'
    # nb_name = '*.ipynb'
    filename = next(glob.iglob(nb_name.format(assignmentNumber)), None)

    print('\n======================= Code Execution =======================\n')

    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         nb_name.format(assignmentNumber), '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ClassDef)):
            # not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *


    
exec_grade = 0

from neuralnetworkA3 import NeuralNetwork

for func in ['NeuralNetwork']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')
        break
    for method in ['_forward', 'get_performance_trace', 'gradient_f',
                    'error_f', 'make_weights_and_views', 'train', 'use']:
        if method not in dir(NeuralNetwork):
            print('CRITICAL ERROR: NeuralNetwork Function named \'{}\' is not defined'.format(method))
            print('  Check the spelling and capitalization of the function name.')
            

print('''\nTesting

    import neuralnetworkA3 as nn

    n_inputs = 3
    n_hiddens = [5, 10, 20]
    n_outputs = 2
    n_samples = 10

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    
    nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
    nnet.all_weights[:] = 0.1  # set all weights to 0.1
    nnet.X_means = np.mean(X, axis=0)
    nnet.X_stds = np.std(X, axis=0)
    nnet.T_means = np.zeros((n_samples, n_outputs))
    nnet.T_stds = np.ones((n_samples, n_outputs))
    
    Y = nnet.use(X)
''')

try:
    pts = 40

    import neuralnetworkA3 as nn

    n_inputs = 3
    n_hiddens = [5, 10, 20]
    n_outputs = 2
    n_samples = 10

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    
    nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
    nnet.all_weights[:] = 0.1  # set all weights to 0.1
    nnet.X_means = np.mean(X, axis=0)
    nnet.X_stds = np.std(X, axis=0)
    nnet.T_means = np.zeros((n_samples, n_outputs))
    nnet.T_stds = np.ones((n_samples, n_outputs))
    
    Y = nnet.use(X)
    
    Y_answer = np.array([[0.14629519, 0.14629519],
                         [0.24029528, 0.24029528],
                         [0.33910878, 0.33910878],
                         [0.43981761, 0.43981761],
                         [0.53920896, 0.53920896],
                         [0.63421852, 0.63421852],
                         [0.72233693, 0.72233693],
                         [0.80186297, 0.80186297],
                         [0.87195874, 0.87195874],
                         [0.93254   , 0.93254   ]])
    
    if np.allclose(Y, Y_answer, 0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct value.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect value. Should be')
        print(Y_answer)
        print(f'        Your value is')
        print(Y)
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor or use raised the exception\n')
    print(ex)




print('''\nTesting
    n_inputs = 3
    n_hiddens = [6, 3]
    n_samples = 5

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    T = np.log(X + 0.1)
    n_outputs = T.shape[1]
    
    def rmse(A, B):
        return np.sqrt(np.mean((A - B)**2))

    results = []
    for rep in range(20):
        nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
        nnet.train(X, T, 5000, 'adam', 0.001, verbose=False)
        Y = nnet.use(X)
        err = rmse(Y, T)
        print(f'Net {rep+1} RMSE {err:.5f}')
        results.append(err)

    mean_rmse = np.mean(results)
    print(mean_rmse)
''')


try:
    pts = 20

    n_inputs = 3
    n_hiddens = [6, 3]
    n_samples = 5

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    T = np.log(X + 0.1)
    n_outputs = T.shape[1]
    
    def rmse(A, B):
        return np.sqrt(np.mean((A - B)**2))

    results = []
    for rep in range(20):
        nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
        nnet.train(X, T, 5000, 'adam', 0.001, verbose=False)
        Y = nnet.use(X)
        err = rmse(Y, T)
        print(f'Net {rep+1} RMSE {err:.5f}')
        results.append(err)

    mean_rmse = np.mean(results)
    print(mean_rmse)
    
    if 0.0 < mean_rmse < 0.1:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct value.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect value. Should be between 0 and 0.1')
        print(f'        Your value is')
        print(mean_rmse)
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor, train, or use raised the exception\n')
    print(ex)


print('''\nTesting
    n_inputs = 3
    n_hiddens = []
    n_samples = 5

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    T = np.log(X + 0.1)
    n_outputs = T.shape[1]
    
    def rmse(A, B):
        return np.sqrt(np.mean((A - B)**2))

    results = []
    for rep in range(20):
        nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
        nnet.train(X, T, 5000, 'adam', 0.001, verbose=False)
        Y = nnet.use(X)
        err = rmse(Y, T)
        print(f'Net {rep+1} RMSE {err:.5f}')
        results.append(err)

    mean_rmse = np.mean(results)
    print(mean_rmse)
''')


try:
    pts = 20

    n_inputs = 3
    n_hiddens = []
    n_samples = 5

    X = np.arange(n_samples * n_inputs).reshape(n_samples, n_inputs) * 0.1
    T = np.log(X + 0.1)
    n_outputs = T.shape[1]
    
    def rmse(A, B):
        return np.sqrt(np.mean((A - B)**2))

    results = []
    for rep in range(20):
        nnet = nn.NeuralNetwork(n_inputs, n_hiddens, n_outputs)
        nnet.train(X, T, 5000, 'adam', 0.001, verbose=False)
        Y = nnet.use(X)
        err = rmse(Y, T)
        print(f'Net {rep+1} RMSE {err:.5f}')
        results.append(err)

    mean_rmse = np.mean(results)
    print(mean_rmse)
    
    if 0.1 < mean_rmse < 0.5:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Returned correct value.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect value. Should be between 0.1 and 0.5')
        print(f'        Your value is')
        print(mean_rmse)
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetwork constructor, train, or use raised the exception\n')
    print(ex)

    



name = os.getcwd().split('/')[-1]

print()
print('='*70)
print('{} Execution Grade is {} / 80'.format(name, exec_grade))
print('='*70)


print('''
___ / 5 Correctly read in bike sharing data using pandas.read_csv.
___ / 5 Correctly ran the required experiments.
___ / 10 Correctly show resulting pandas DataFrame and discuss results.''')

print()
print('='*70)
print('{} Experiments and Discussion Grade is __ / 20'.format(name))
print('='*70)



print()
print('='*70)
print('{} FINAL GRADE is  _  / 100'.format(name))
print('='*70)

print('''
Extra Credit:  Code and discussion showing most significant input features.
''')

print('\n{} EXTRA CREDIT is 0 / 1'.format(name))

if run_my_solution:
    # print('##############################################')
    # print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    # print('##############################################')
    pass

