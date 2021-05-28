import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch.cuda
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tools import tools_parse_eval_file
import random
import math

random.seed(667)

def calcuate_wenjuanxing():
    text = """
    2	3	2	2	2	4	4	4	3	4	1	1x
    7	10	4	7	4	7	8	8	7	6	2	3x
    5	8	3	6	5	8	6	8	5	8	1	4x
    2	2	2	2	2	4	3	3	2	2	2	3x
    8	7	10	9	10	10	9	9	10	10	10	10x
    3	3	3	1	3	5	6	2	4	7	1	2x
    3	6	2	4	3	5	5	2	7	6	5	5x
    3	8	1	2	2	7	2	6	4	10	1	3x
    7	6	7	5	7	10	10	10	9	5	7	5x
    4	8	5	7	4	8	8	8	6	8	4	6x
    2	6	1	3	1	2	1	2	1	2	1	1x
    5	4	3	3	3	5	4	3	5	5	2	3x
    2	7	2	3	3	8	4	8	3	8	2	4x
    1	4	1	2	2	4	1	3	3	8	1	5x
    1	3	1	1	2	2	3	5	3	8	1	1x
    1	6	1	5	2	5	2	4	2	6	1	4x
    1	5	1	2	1	3	5	5	4	5	1	1x
    1	7	10	6	1	7	1	5	1	8	1	3x
    5	6	5	7	6	8	3	3	3	3	3	4x
    1	8	1	5	1	7	5	10	2	10	2	5x
    2	2	2	2	4	3	3	2	3	2	4	3x
    2	7	2	5	1	7	3	8	1	9	2	4x
    3	8	2	2	2	1	1	6	1	7	1	1x
    5	5	3	4	3	4	8	3	3	6	3	2x
    3	7	2	4	3	5	3	8	2	5	2	4x
    1	3	1	1	1	1	1	4	1	4	1	4x
    4	9	2	6	7	9	5	10	7	10	6	8x
    3	8	2	6	2	4	2	7	2	7	1	5x
    1	2	1	1	1	1	1	2	1	2	1	1x
    """
    text = text.split('x')
    score_my = {'nature': {0: [], 1: [], 2: []}, 'consist': {0: [], 1: [], 2: []}}
    score_cteg = {'nature': {0: [], 1: [], 2: []}, 'consist': {0: [], 1: [], 2: []}}
    num = 0
    for line in text:
        line = line.strip('\n').strip().split('\t')
        if len(line) != 12: continue
        line = list(map(float, line))
        num += 1
        # cnt += 1
        # print(cnt, line)
        i = 0
        # our: 0 1 6 7 8 9
        # cte: 2 3 4 5 10 11
        cmy = 0
        cct = 0
        while i < len(line):
            if i in {0, 6, 8}:
                score_my['nature'][cmy].append(line[i])
                score_my['consist'][cmy].append(line[i + 1])
                cmy += 1
                i += 2
            else:
                score_cteg['nature'][cct].append(line[i])
                score_cteg['consist'][cct].append(line[i + 1])
                cct += 1
                i += 2

    # min-max normalize
    eps = 1e-12
    for i in range(num):
        name = 'nature'
        n_min = 1e9
        n_max = -1e9
        for j in range(3):
            n_min = min(score_my[name][j][i], n_min)
            n_min = min(score_cteg[name][j][i], n_min)
            n_max = max(score_my[name][j][i], n_max)
            n_max = max(score_cteg[name][j][i], n_max)
        # if n_max == n_min:
        #     print(n_max, n_min, i)
        #     exit(0)
        for j in range(3):
            score_my[name][j][i] = (score_my[name][j][i] - n_min) / (n_max - n_min + eps) * 10 + 1
            score_cteg[name][j][i] = (score_cteg[name][j][i] - n_min) / (n_max - n_min + eps) * 10 + 1

        name = 'consist'
        n_min = 1e9
        n_max = -1e9
        for j in range(3):
            n_min = min(score_my[name][j][i], n_min)
            n_min = min(score_cteg[name][j][i], n_min)
            n_max = max(score_my[name][j][i], n_max)
            n_max = max(score_cteg[name][j][i], n_max)
        # if n_max == n_min:
        #     print(n_max, n_min)
        #     exit(0)
        for j in range(3):
            score_my[name][j][i] = (score_my[name][j][i] - n_min) / (n_max - n_min + eps) * 10 + 1
            score_cteg[name][j][i] = (score_cteg[name][j][i] - n_min) / (n_max - n_min + eps) * 10 + 1

    # output mean
    advance_nature, advance_consist = 0.0, 0.0
    for i in range(3):
        cteg_n = sum(score_cteg['nature'][i]) / num
        my_n = sum(score_my['nature'][i]) / num
        cteg_c = sum(score_cteg['consist'][i]) / num
        my_c = sum(score_my['consist'][i]) / num
        print(f'num {i + 1} cteg-my\nnature: {cteg_n:.2f}--{my_n:.2f}\nconsis: {cteg_c:.2f}--{my_c:.2f}')
        print('--' * 10)
        advance_consist += (my_c - cteg_c) / cteg_c
        advance_nature += (my_n - cteg_n) / cteg_n

    advance_consist /= 3
    advance_nature /= 3
    print(f'nature lead {advance_nature:.5f} consistency lead {advance_consist:.5f}')


train_loss, test_loss, novelty, div1, div2, bleu2, mixbleu4 = {}, {}, {}, {}, {}, {}, {}
ppl = {}
datas = {}


def show_chapter_main():
    pass
    # show(epoch, train_loss['simple'], train_loss['cteg'], train_loss['ours'], savepath='figures/train_loss.pdf')
    # show(1.5, 8.0, 'train loss', 59, train_loss['simple'], savepath='figures/pre-trained_loss.pdf', location='upper right')
    # exit(0)

    # name = 'test_loss'
    # temp = datas[name]
    # show(6.5, 13, 'ln(PPL)', epoch, temp['cteg'], temp['ours'], savepath=f'figures/lnppl.pdf',
    #      sample=30, location='lower right')

    # name = 'train_loss'
    # temp = datas[name]
    # show(1.5, 8.0, 'train loss', epoch, temp['cteg'], temp['ours'], savepath=f'figures/{name}.pdf',
    #      sample=None, location='upper right')
    #
    # name = 'bleu2'
    # temp = datas[name]
    # show(0.0, 0.092, 'BLEU-2', epoch,  temp['cteg'], temp['ours'], savepath=f'figures/{name}.pdf',
    #      sample=30)
    #
    # name = 'bleu4'
    # temp = datas[name]
    # show(0.0, 0.034, 'BLEU-4', epoch,  temp['cteg'], temp['ours'], savepath=f'figures/{name}.pdf',
    #      sample=30)
    #
    # name = 'div1'
    # temp = datas[name]
    # show(0.0, 0.065, 'Diversity-1', epoch,  temp['cteg'], temp['ours'], savepath=f'figures/{name}.pdf',
    #      sample=30)
    #
    # name = 'div2'
    # temp = datas[name]
    # show(0.0, 0.38, 'Diversity-2', epoch,  temp['cteg'], temp['ours'], savepath=f'figures/{name}.pdf',
    #      sample=30)
    #
    # name = 'novelty'
    # temp = datas[name]
    # show(0.75, 0.92, 'Novelty', epoch,  temp['cteg'], temp['ours'], savepath=f'figures/{name}.pdf',
    #      sample=30)

def show_ablation_dropout():
    pass
    # my_div_variation = [0, 0]
    # cteg_div_variation = [0, 0]
    # cnt = 0
    # for i in range(60, 101):
    #     cnt += 1
    #     my_div_variation[0] += (datas['div1']['ours'][i] - datas['div1']['ours_nodrop'][i]) / datas['div1']['ours_nodrop'][i]
    #     my_div_variation[1] += (datas['div2']['ours'][i] - datas['div2']['ours_nodrop'][i]) / datas['div2']['ours_nodrop'][i]
    #     cteg_div_variation[0] += (datas['div1']['cteg'][i] - datas['div1']['cteg_nodrop'][i]) / datas['div1']['cteg_nodrop'][i]
    #     cteg_div_variation[1] += (datas['div2']['cteg'][i] - datas['div2']['cteg_nodrop'][i]) / datas['div2']['cteg_nodrop'][i]
    #
    # my_div_variation[0] /= cnt
    # my_div_variation[1] /= cnt
    # cteg_div_variation[0] /= cnt
    # cteg_div_variation[1] /= cnt
    # variation = {
    #     'ours': my_div_variation, 'cteg': cteg_div_variation
    # }
    #
    # for i in range(2):
    #     name = f'div{i+1}'
    #     for model in ['ours', 'cteg']:
    #         drop = sum(datas[name][model][60:101]) / cnt
    #         no =  sum(datas[name][f'{model}_nodrop'][60:101]) / cnt
    #
    #         print(f"{name} {model} drop {drop:.5f} no {no:.5f} {variation[model][i]:.5f}")

    # my_nov_variation = 0
    # cteg_nov_variation = 0
    # name = 'bleu4'
    # cnt = 0
    # temp = datas[name]
    # for i in range(60, 101):
    #     cnt += 1
    #     my_nov_variation += (temp['ours'][i] - temp['ours_nodrop'][i]) / temp['ours_nodrop'][i]
    #     cteg_nov_variation += (temp['cteg'][i] - temp['cteg_nodrop'][i]) / temp['cteg_nodrop'][i]
    #
    #
    # my_nov_variation /= cnt
    # cteg_nov_variation /= cnt
    # variation = {
    #     'ours': my_nov_variation, 'cteg': cteg_nov_variation
    # }
    #
    #
    #
    # for model in ['ours', 'cteg']:
    #     drop = sum(temp[model][60:101]) / cnt
    #     no = sum(temp[f'{model}_nodrop'][60:101]) / cnt
    #     print(f"{name} {model} drop {drop:.5f} no {no:.5f} {variation[model]:.5f}")


    # figures above ***************************************************************************************
    # sample = 20
    #
    #
    # name = 'bleu4'
    # temp = datas[name]
    # show(0.0, 0.035, 'BLEU-4', epoch,  temp['cteg'], temp['ours'], temp['cteg_nodrop'], temp['ours_nodrop'],
    #      savepath=f'figures/ab_{name}.pdf', sample=sample)
    #
    # name = 'div1'
    # temp = datas[name]
    # show(0.0, 0.065, 'Diversity-1', epoch, temp['cteg'], temp['ours'], temp['cteg_nodrop'], temp['ours_nodrop'],
    #      savepath=f'figures/ab_{name}.pdf', sample=sample)
    # #
    # name = 'div2'
    # temp = datas[name]
    # show(0.0, 0.38, 'Diversity-1', epoch, temp['cteg'], temp['ours'], temp['cteg_nodrop'], temp['ours_nodrop'],
    #      savepath=f'figures/ab_{name}.pdf', sample=sample)
    # #
    # name = 'novelty'
    # temp = datas[name]
    # show(0.75, 0.92, 'Novelty', epoch, temp['cteg'], temp['ours'], temp['cteg_nodrop'], temp['ours_nodrop'],
    #      savepath=f'figures/ab_{name}.pdf', sample=sample)

def show_ablation_heads():
    pass
# name = 'bleu2'
    # temp = datas[name]
    # show(0.0, 0.09, 'BLEU-2', epoch,  temp['ours'], temp['ours_nopre'], savepath=f'figures/ab_pre_{name}.pdf', sample=sample)

    # name = 'div1'
    # temp = datas[name]
    # show(0.0, 0.05, 'Diversity-1', epoch, temp['ours'], temp['ours_nopre'], savepath=f'figures/ab_pre_{name}.pdf', sample=sample)
    # exit(0)
    # #
    # name = 'train_loss'
    # temp = datas[name]
    # show(0.6, 8.0, 'train loss', epoch, temp['ours'], temp['ours_nopre'], savepath=f'figures/ab_pre_{name}.pdf',
    #      sample=sample, location='upper right')
    # #
    # name = 'novelty'
    # temp = datas[name]
    # show(0.75, 0.92, 'Novelty', epoch, temp['ours'], temp['ours_nopre'], savepath=f'figures/ab_pre_{name}.pdf', sample=sample)
def plot_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def show_attention():
    pass
    # too many works, give up *******************************************5/26****************
    # from model_builder import build_model
    # from data import read_acl_origin_data, ZHIHU_dataset
    # from config import config_zhihu_dataset, config_concepnet
    # from tools import tools_setup_seed
    # tools_setup_seed(667)
    # device = torch.device('cuda:3')
    # load_path = 'logs/knowledge/21-05-22-15_01_32/model_state/epoch_74_21-05-22-23_40_01.pt'
    # topic = "生活 性格 <pad> <pad> <pad>".split(' ')
    # text = "我 把 谁 都 当 朋友 ， 和 谁 都 掏 心窝子 ， 导致 说 了 很多 不 该 说 的 话 。 如今 已经 踏 入 社会 ， 这种 性格 急需 改正 ， 此前 也 都 因为 这 张 嘴 吃 过 亏 ， 但 就是 不 长 记性 ， 有 没 有 什么 办法 能够 改正 ？ 让 自己 变 得 话 少 ， 认清 什么 场合 说 什么样 的 话 ， 什么 事 能 告诉 什么样 的 人 ？".split(' ')
    # word2idx, idx2word, topic2idx, idx2topic, (train_essay, train_topic, train_mem), (
    #     test_essay, test_topic, test_mem) \
    #     = read_acl_origin_data()
    # train_all_dataset = ZHIHU_dataset(path=config_zhihu_dataset.train_data_path,
    #                                   topic_num_limit=config_zhihu_dataset.topic_num_limit,
    #                                   topic_padding_num=config_zhihu_dataset.topic_padding_num,
    #                                   vocab_size=config_zhihu_dataset.vocab_size,
    #                                   essay_padding_len=config_zhihu_dataset.essay_padding_len,
    #                                   prior=None, encode_to_tensor=True,
    #                                   mem_corpus_path=config_concepnet.memory_corpus_path,
    #                                   acl_datas=(word2idx, idx2word, topic2idx, idx2topic,
    #                                              (train_essay, train_topic, train_mem)))
    #
    # essay_input, essay_target, real_len = train_all_dataset.convert_word2idx(text)
    # topic_tensor = None
    # seq2seq = build_model(model_name='knowledge', dataset_name='acl',
    #                       vocab_size=50004, device=device,
    #                       load_path=load_path, init_way='normal',
    #                       mask_idx=word2idx['<pad>'])

def show(blow, bhigh, ylabel, epoch:int, *args, savepath=None, sample=None, location='lower right'):
    # 设置图片大小
    # plt.figure(figsize=(15, 20), dpi=300)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 画图——折线图
    plt.clf()
    args = [a[:epoch] for a in args]
    if sample:
        step = epoch // sample
        temp = [i for i in range(0, epoch, step)]
        if epoch % step != 0:
            temp.append(epoch-1)
        epoch = temp
        args = [
            [a[i] for i in epoch] for a in args
        ]
    else:
        epoch = [i for i in range(epoch)]

    # plt.plot(epoch, args[0], label='pre-trained', color="r", marker='x', linewidth=1, markersize=5)
    # plt.plot(epoch, args[0],  label='CTEG', color="g", marker='v', linewidth=1, markersize=5)
    plt.plot(epoch, args[0],  label='Ours', color="r", marker='o', linewidth=1, markersize=5)
    # plt.plot(epoch, args[2], label='CTEG*', color="y", marker='^', linewidth=1, markersize=5)
    plt.plot(epoch, args[1], label='Ours*', color="b", marker='x', linewidth=1, markersize=5)
    # plt.plot(datas[0], datas[4],  label='test_acc', color="g", marker='*')
    # plt.plot(datas[0], datas[5],  label='test_acc', color="deeppink", marker='d')

    # 设置网格线
    plt.grid(alpha=0.2)
    plt.legend(loc=location)
    # plt.title("result")
    # my_y_ticks = np.arange(0, 1, 0.01)
    # plt.yticks(my_y_ticks)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    bound_low = blow
    bound_high = bhigh
    plt.ylim(bound_low, bound_high)
    x_ticks = np.linspace(bound_low, bound_high, 8)  # 产生区间在-5至4间的10个均匀数值
    plt.yticks(x_ticks)  # 将linspace产生的新的十个值传给xticks( )函数，用以改变坐标刻度
    ax = plt.gca()

    plt.savefig(savepath, dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.show()

def add_eval_file(path, name, epoch_limit=150):
    _, trl, tel, nv, d1, d2, b2, m4 = tools_parse_eval_file(path)
    train_loss[name] = trl[:epoch_limit]
    test_loss[name] = tel[:epoch_limit]
    novelty[name] = nv[:epoch_limit]
    div1[name] = d1[:epoch_limit]
    div2[name] = d2[:epoch_limit]
    bleu2[name] = b2[:epoch_limit]
    mixbleu4[name] = m4[:epoch_limit]
    ppl[name] = [math.exp(x) for x in test_loss[name]]





if __name__ == '__main__':

    # calcuate_wenjuanxing()
    # exit(0)
    datas['train_loss'] = train_loss
    datas['test_loss'] = test_loss
    datas['ppl'] = ppl
    datas['novelty'] = novelty
    datas['div1'] = div1
    datas['div2'] = div2
    datas['bleu2'] = bleu2
    datas['bleu4'] = mixbleu4

    epoch = 150

    loads = {
        'ours' : 'logs/knowledge/21-05-22-15_01_32/evaluate.log',
        'ours_nodrop' : 'logs/knowledge/21-05-21-09_49_38/evaluate.log',
        'ours_nopre' : 'logs/knowledge/21-05-23-10_19_06/evaluate.log',
        'cteg' : 'logs/cteg/21-05-14-09_00_28_using_activate_dropout_in_eval/evaluate.log',
        'cteg_nodrop' : 'logs/cteg/21-05-13-15_50_05_no_dropout_in_eval/evaluate.log'
    }

    for k, v in loads.items(): add_eval_file(v, k, epoch_limit=epoch)

    # figures above ***************************************************************************************
    sample = 20


















