import string
import re


def pre_process(text):
    text = re.sub(
        r'([a-zA-ZáàâãéêíóõôúçÁÀÂÃÉÊÍÓÕÔÚÇ, ])\n', r'\1.\n', text)
    text = re.sub(r'[0-9"-,()ºª;$€&´`]+', '', text)
    text = text.replace('americano\n', 'americano.')
    text = text.replace('\n', ' ')
    return text


def filter_list(l):
    return list(filter(lambda s: s not in list(string.punctuation) + ['', ' ', '\n', '“', '``', '’'], l))
