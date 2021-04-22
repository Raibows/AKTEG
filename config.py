

class config_train:
    epoch = 30
    batch_size = 64
    learning_rate = 1e-3
    fold_k = 5

class config_zhihu_dataset:
    device_name = 'cuda:0'
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
    pretrained_wv_path = './zhihu_dataset/zhihu.pretrained.wv'
    pretrained_wv_dim = 300
    essay_preprocess_wv_path = './zhihu_dataset/zhihu.pretrained.train_essay.wv'
    topic_preprocess_wv_path = './zhihu_dataset/zhihu.pretrained.train_topic.wv'
    memory_preprocess_wv_path = './zhihu_dataset/zhihu.pretrained.train_memory.wv'
    topic_synonym_normal_num = 20
    topic_synonym_max_num = topic_synonym_normal_num * topic_padding_num
    memory_special_tokens = {'oov': 0}
    topic_2_mems_corpus_path = './zhihu_dataset/train_dataset.memory'
    mem2idx_and_idx2mem_path = './zhihu_dataset/train_dataset.memory.dict'



class config_seq2seq:
    model_save_fmt = './results/seq2seq/{}/test_loss{:.5f}.pt'
    model_load_path = None
    encoder_embed_size = 300
    encoder_lstm_layer_num = 2
    encoder_lstm_hidden_size = 300
    encoder_lstm_is_bid = True

    decoder_embed_size = 300
    decoder_lstm_layer_num = encoder_lstm_layer_num

    teacher_force_rate = 0.75
