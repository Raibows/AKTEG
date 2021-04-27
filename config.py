

class config_train:
    device_name = 'cuda:3'
    epoch = 150
    batch_size = 128
    learning_rate = 1e-3
    fold_k = 1
    dataloader_num_workers = 4
    is_load_model = False
    is_save_model = True
    grad_clip_norm_type = 2.0
    grad_clip_max_norm = 1.0
    model_init_way = 'noraml'

class config_concepnet:
    raw_path = './concepnet/chineseconceptnet.csv'
    reserved_data_path = './concepnet/reserved.dict.pkl'
    memory_corpus_path = './concepnet/memory_corpus.pkl'


class config_zhihu_dataset:
    raw_data_path = './zhihu_dataset/raw.txt'
    train_data_path = './zhihu_dataset/train.txt'
    test_data_path = './zhihu_dataset/test.txt'
    # coco_test_path = './image_coco/image_coco_test.txt'
    # coco_train_path = './image_coco/image_coco.txt'

    test_data_split_ratio = 1 / 6
    topic_num_limit = 100
    vocab_size = 50000
    remove_high_freq_top = 0
    topic_padding_num = 5
    essay_padding_len = 80
    pretrained_wv_path = {
        'tencent': './zhihu_dataset/pretrained_embeddings/tencent.pretrained.wv',
        'zhihu': './zhihu_dataset/pretrained_embeddings/zhihu.pretrained.wv'
    }
    pretrained_wv_dim = {
        'tencent': 200,
        'zhihu': 300
    }
    topic_mem_per_num = 24
    topic_mem_max_num = topic_mem_per_num * topic_padding_num
    special_tokens = {'<pad>', '<go>', '<eos>', '<unk>'}

    preprocess_topic_threshold = 2
    preprocess_essay_min_len = essay_padding_len // 1.8



class config_seq2seq:
    model_save_fmt = './results/seq2seq/{}/epoch_{}_test_loss_{:.5f}.pt'
    model_load_path = None
    encoder_lstm_hidden_size = 512
    encoder_lstm_is_bid = True
    lstm_layer_num = 1

    attention_size = 128
    teacher_force_rate = 1.0

    embedding_size = 200
    pretrained_wv_path = {
        'tencent': './zhihu_dataset/tencent.seq2seq.wv',
        'zhihu': './zhihu_dataset/zhihu.seq2seq.wv'
    }

