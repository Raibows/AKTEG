
class config_zhihu_dataset:
    raw_data_path = './zhihu_dataset/raw.txt'
    train_data_path = './zhihu_dataset/train.txt'
    test_data_path = './zhihu_dataset/test.txt'
    test_data_split_ratio = 1 / 6
    topic_num_limit = 100
    essay_vocab_size = 50000
    topic_threshold = 3
    topic_padding_num = 5
    essay_padding_len = 100
    topic_special_tokens = {'<pad_topic>': 0, '<unk_topic>': 1, '<fake_topic>': 2}
    essay_special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3, }