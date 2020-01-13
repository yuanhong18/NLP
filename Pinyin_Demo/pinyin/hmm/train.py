# -*- coding=utf8 -*-
"""
    获取HMM模型
"""
from __future__ import division
from tqdm import tqdm
from math import log

from pypinyin import pinyin, NORMAL
from pinyin.model import (
    Transition,
    Emission,
    Starting,
    init_hmm_tables,
    HMMSession
)
from pinyin.utils import iter_dict


def init_start():
    """
    初始化起始概率
    """
    freq_map = {}
    total_count = 0
    print("Count start frequence:")
    for phrase, frequency in tqdm(iter_dict()):
        total_count += frequency
        freq_map[phrase[0]] = freq_map.get(phrase[0], 0) + frequency
    print("Calculate start frequence:")
    for character, frequency in tqdm(freq_map.items()):
        Starting.add(character, log(frequency / total_count))


def init_emission():
    """
    初始化发射概率
    """
    character_pinyin_map = {}
    print("Count emission frequence:")
    for phrase, frequency in tqdm(iter_dict()):
        pinyins = pinyin(phrase, style=NORMAL)
        for character, py in zip(phrase, pinyins):
            character_pinyin_count = len(py)
            if character not in character_pinyin_map:
                character_pinyin_map[character] = \
                    {x: frequency/character_pinyin_count for x in py}
            else:
                pinyin_freq_map = character_pinyin_map[character]
                for x in py:
                    pinyin_freq_map[x] = pinyin_freq_map.get(x, 0) + \
                                         frequency/character_pinyin_count
    print("Calculate emission frequence:")
    for character, pinyin_map in tqdm(character_pinyin_map.items()):
        sum_frequency = sum(pinyin_map.values())
        for py, frequency in pinyin_map.items():
            Emission.add(character, py, log(frequency/sum_frequency))


def init_transition():
    """
    初始化转移概率
    """
    # todo 优化 太慢
    transition_map = {}
    print("Count transition frequence:")
    for phrase, frequency in tqdm(iter_dict()):
        for i in range(len(phrase) - 1):
            if phrase[i] in transition_map:
                transition_map[phrase[i]][phrase[i+1]] = \
                    transition_map[phrase[i]].get(phrase[i+1], 0) + frequency
            else:
                transition_map[phrase[i]] = {phrase[i+1]: frequency}
    print("Calculate transition frequence:")
    for previous, behind_map in tqdm(transition_map.items()):
        sum_frequency = sum(behind_map.values())
        for behind, freq in behind_map.items():
            Transition.add(previous, behind, log(freq / sum_frequency))


if __name__ == '__main__':
    print("Init hmm table...")
    init_hmm_tables()
    print("Init start...")
    init_start()
    print("Init emission...")
    init_emission()
    print("Init transition...")
    init_transition()

    # 创建索引
    session = HMMSession()
    session.execute('create index ix_starting_character on starting(character);')
    session.execute('create index ix_emission_character on emission(character);')
    session.execute('create index ix_emission_pinyin on emission(pinyin);')
    session.execute('create index ix_transition_previous on transition(previous);')
    session.execute('create index ix_transition_behind on transition(behind);')
    session.commit()
