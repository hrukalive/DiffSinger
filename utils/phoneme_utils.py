import pathlib

try:
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
except ModuleNotFoundError:
    rank_zero_info = print

_initialized = False
_ALL_CONSONANTS_SET = set()
_ALL_VOWELS_SET = set()
_dictionary = {
    'AP': ['AP'],
    'SP': ['SP']
}
_phoneme_list: list


def locate_dictionary(config: dict):
    """
    Search and locate the dictionary file.
    Order:
    1. config['dictionary']
    2. config['g2p_dictionary']
    3. 'dictionary.txt' in config['work_dir']
    4. file with same name as config['g2p_dictionary'] in config['work_dir']
    :return: pathlib.Path of the dictionary file
    """
    assert 'dictionary' in config or 'g2p_dictionary' in config, \
        'Please specify a dictionary file in your config.'
    config_dict_path = pathlib.Path(config['dictionary'])
    if config_dict_path.exists():
        return config_dict_path
    work_dir = pathlib.Path(config['work_dir'])
    ckpt_dict_path = work_dir / config_dict_path.name
    if ckpt_dict_path.exists():
        return ckpt_dict_path
    ckpt_dict_path = work_dir / 'dictionary.txt'
    if ckpt_dict_path.exists():
        return ckpt_dict_path
    raise FileNotFoundError('Unable to locate the dictionary file. '
                            'Please specify the right dictionary in your config.')


def _build_dict_and_list(config: dict):
    global _dictionary, _phoneme_list

    _set = set()
    with open(locate_dictionary(config), 'r', encoding='utf8') as _df:
        _lines = _df.readlines()
    for _line in _lines:
        _pinyin, _ph_str = _line.strip().split('\t')
        _dictionary[_pinyin] = _ph_str.split()
    for _list in _dictionary.values():
        [_set.add(ph) for ph in _list]
    _phoneme_list = sorted(list(_set))
    rank_zero_info('| load phoneme set: ' + str(_phoneme_list))


def _initialize_consonants_and_vowels():
    # Currently we only support two-part consonant-vowel phoneme systems.
    for _ph_list in _dictionary.values():
        _ph_count = len(_ph_list)
        if _ph_count == 0 or _ph_list[0] in ['AP', 'SP']:
            continue
        elif len(_ph_list) == 1:
            _ALL_VOWELS_SET.add(_ph_list[0])
        else:
            _ALL_CONSONANTS_SET.add(_ph_list[0])
            _ALL_VOWELS_SET.add(_ph_list[1])


def _initialize(config: dict):
    global _initialized
    if not _initialized:
        _build_dict_and_list(config)
        _initialize_consonants_and_vowels()
        _initialized = True


def get_all_consonants(config: dict):
    _initialize(config)
    return sorted(_ALL_CONSONANTS_SET)


def get_all_vowels(config: dict):
    _initialize(config)
    return sorted(_ALL_VOWELS_SET)


def build_dictionary(config: dict) -> dict:
    _initialize(config)
    return _dictionary


def build_phoneme_list(config: dict) -> list:
    _initialize(config)
    return _phoneme_list
