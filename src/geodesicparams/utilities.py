from sympy import re, im
from os import path, unlink, walk
from shutil import rmtree

def inlist(element, lst):
    for i in range(len(lst)):
        if element == lst[i]:
            result = i
            break
        else:
            result = -1
    return result

def extract_multiple_elems(lst):
    sorted_lst = sorted(lst, key = lambda x : re(x))
    mult_elems = []
    clean_lst = sorted_lst
    i = 0
    while i < len(sorted_lst):
        if i == len(sorted_lst) - 1:
            break
        else:
            if sorted_lst[i] == sorted_lst[i+1]:
                mult_elems.append(sorted_lst[i])
                clean_lst.pop(i)
            else:
                i += 1
    return clean_lst, mult_elems

def find_next(expression, lst):
    if type(expression) == list:
        expression = expression[0]
    d = abs(expression - lst[0])
    j = 0
    for i in range(1, len(lst)):
        if abs(expression-lst[i]) < d:
            d = abs(expression - lst[i])
            j = i
    return lst[j]

def separate_zeros(zeros):
    realNS = []
    complexNS = []
    for i in range(len(zeros)):
        if im(zeros[i]) == 0:
            realNS.append(zeros[i])
        else:
            complexNS.append(zeros[i])
    realNS.sort()
    return realNS, complexNS

def eval_roots(lst):
    eval_lst = [lst[i].evalf() for i in range(len(lst))]
    return eval_lst

def clear_directory(dir):
    for root, dirs, files in walk(dir):
        for f in files:
            unlink(path.join(root, f))
        for d in dirs:
            rmtree(path.join(root, d))
