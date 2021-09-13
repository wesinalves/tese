import os

models = [
        'beijing10',
        'beijing20',
        'beijing30',
        'beijing40',
        'beijing50',
        'beijing60',
        'beijing70',
        'beijing80',
        'beijing90',
        ]

for model in models:
    os.system(f'python code/test.py data {model} --input channel')