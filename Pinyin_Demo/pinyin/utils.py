# -*- coding=utf8 -*-
import os

dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dict.txt')


def iter_dict():
    """
    遍历dict.txt文件
    """
    with open(dict_path, 'r', encoding="utf-8") as f:
        for line in f:
            phrase, frequency, tag = line.split()
            yield phrase, int(frequency)
