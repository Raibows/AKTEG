

class config_train:
    device_name = 'cuda:0'
    epoch = 20
    batch_size = 160
    learning_rate = 1e-4
    fold_k = 1
    dataloader_num_workers = 4
    is_load_model = False
    is_save_model = False
    grad_clip_norm_type = 2.0
    grad_clip_max_norm = 1.0

class config_concepnet:
    raw_path = './concepnet/chineseconceptnet.csv'
    reserved_data_path = './concepnet/reserved.dict.pkl'
    memory_pretrained_wv_path = './concepnet/zhihu.pretrained.train_memory.wv'
    topic_2_mems_corpus_path = './concepnet/train_dataset.memory'
    mem2idx_and_idx2mem_path = './concepnet/train_dataset.memory.dict'
    memory_special_tokens = {'oov': 0}


class config_zhihu_dataset:
    raw_data_path = './zhihu_dataset/raw.txt'
    train_data_path = './zhihu_dataset/train.txt'
    test_data_path = './zhihu_dataset/test.txt'
    test_data_split_ratio = 1 / 6
    topic_num_limit = 100
    essay_vocab_size = 40000
    topic_threshold = 3
    remove_high_freq_top = 30
    topic_padding_num = 6 # 5 to 6 is for eos
    essay_padding_len = 80
    topic_special_tokens = {'<pad_topic>': 0, '<eos_topic>': 1, '<unk_topic>': 2, '<fake_topic>': 3}
    essay_special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    pretrained_wv_path = './zhihu_dataset/zhihu.pretrained.wv'
    pretrained_wv_dim = 300
    # topic_preprocess_wv_path = None
    topic_mem_normal_num = 20
    topic_mem_max_num = topic_mem_normal_num * topic_padding_num


class config_seq2seq:
    model_save_fmt = './results/seq2seq/{}/test_loss{:.5f}.pt'
    model_load_path = './results/seq2seq/21-04-23-12_31_54/test_loss6.84044.pt'
    encoder_lstm_layer_num = 1
    encoder_lstm_hidden_size = 64
    encoder_lstm_is_bid = True

    decoder_lstm_layer_num = encoder_lstm_layer_num

    attention_size = 64
    teacher_force_rate = 1.0

    embedding_size = 300
    pretrained_wv_path = './zhihu_dataset/seq2seq.wv'

