## Attention based Knowledge enhanced Topic-to-Essay Generation, AKTEG
TEG aims at learning the process of given several topics and generating a paragraph­level, meaningful, diverse and novel essay with contents closely related to the given topics.

This repository contains an Attention based knowledge enhanced model which mainly uses Multi-Head Attention mechanism to extract the additional information from the external commonsense knowledge to integrate to the essay generation process.

## Prepare
We used a prepared dataset which can be downloaded from <a href="https://pan.baidu.com/s/17pcfWUuQTbcbniT0tBdwFQ">CTEG</a>.

## Run
Using command below to train our model and evaluate with every epochs.
```
python train_generator.py \
--model attention \
--device 0 \
--dataset acl \
--epoch 150 \
--batch 64
```

Get more details by ``-h``.

## Reference
<a href="https://github.com/libing125/CTEG">Github / CTEG</a>

<a href="https://www.aclweb.org/anthology/P19-1193">ACL19 / Enhancing Topic-to-Essay Generation with External Commonsense Knowledge</a>