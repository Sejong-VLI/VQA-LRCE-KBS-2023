# Lightweight Recurrent Cross-modal Encoder

## Setup and Configurations
### Environment
Install all the dependencies using conda environment by typing:
```
conda env create -f env.yaml
conda activate lrce
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
```

### Dataset
#### MSVD-QA
Download the <a href="https://mega.nz/#!QmxFwBTK!Cs7cByu_Qo42XJOsv0DjiEDMiEm8m69h60caDYnT_PQ">annotations</a> and <a href="https://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar">videos</a>. Extract them into a single directory and place all of the videos under a folder named `video`. Download the <a href="https://sejonguniversity-my.sharepoint.com/:u:/g/personal/22110338_sju_ac_kr/EX4hOi2kqxRIpaJU9uHTE48BIi0XNM1CCuS_I9Cyt8LkUA?e=OxGvay">idx-video-mapping.pkl</a> and place it on the same directory. The dataset directory should look as follows:
```
MSVD-QA
├── idx-video-mapping.pkl
├── readme.txt
├── test_qa.json
├── train_qa.json
├── val_qa.json
└── video
    ├── 00jrXRMlZOY_0_10.avi
    ├── 02Z-kuB3IaM_2_13.avi
    ...
    └── zzit5b_-ukg_5_20.avi
``` 


#### MSRVTT-QA
Download the <a href="https://mega.nz/#!UnRnyb7A!es4XmqsLxl-B7MP0KAat9VibkH7J_qpKj9NcxLh8aHg">annotations</a> and <a href="https://www.mediafire.com/folder/h14iarbs62e7p/shared">videos</a>. Extract them into a single directory and place all of the videos under a folder named `video`. Download the <a href="https://sejonguniversity-my.sharepoint.com/:u:/g/personal/22110338_sju_ac_kr/EbIhKUEt5xFAiZim63043wYBIdSstocByGSfC9vK5xOihA?e=FsrrbS">idx-video-mapping.pkl</a> and place it on the same directory. The dataset directory should look as follows:
```
MSRVTT-QA
├── category.txt
├── idx-video-mapping.pkl
├── readme.txt
├── test_qa.json
├── train_qa.json
├── val_qa.json
└── video
    ├── video0.mp4
    ├── video1000.mp4
    ...
    └── video9.mp4
```
#### TGIF-QA
Download the annotations and gifs from the <a href="https://github.com/YunseokJANG/tgif-qa">official repo</a>. Combine all the files into a single directory and restructure it as follows:
```
TGIF-QA
├── annotations
│   ├── README.md
│   ├── Test_action_question.csv
│   ├── Test_count_question.csv
│   ├── Test_frameqa_question.csv
│   ├── Test_transition_question.csv
│   ├── Total_action_question.csv
│   ├── Total_count_question.csv
│   ├── Total_frameqa_question.csv
│   ├── Total_transition_question.csv
│   ├── Train_action_question.csv
│   ├── Train_count_question.csv
│   ├── Train_frameqa_question.csv
│   └── Train_transition_question.csv
└── gifs
    ├── tumblr_ku4lzkM5fg1qa47qco1_250.gif
    ├── tumblr_ky2syrOMmW1qawjc8o1_250.gif
    ...
    └── tumblr_nrlo5nKKip1uz642so1_400.mp4
```
### GPU
This code will utilize all of the GPU in your machine by default. To only use some of the GPUs, you can set the `CUDA_VISIBLE_DEVICES` variable in your environment. For example, to use only the first GPU, type:
```
export CUDA_VISIBLE_DEVICES=0
```

### Feature Extractor
Download the pre-trained video swin transformer <a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth">here</a>. Then, place it under the `pretrained_models` directory of this project.

## Performance 
![performance](https://i.imgur.com/MhEJwgf.png)
## Training
To see all of the possible arguments and their explanation when performing training, you can type:
```
python train.py -h
```

We provided the arguments that we used to reproduce the reported performance in the paper as follows:
- MSVD-QA
```
python train.py --dataset msvd-qa-oe \
--dataset-dir <path/to/dataset> --ckpt-interval 2 --batch-size 10 \
--epoch 8 --drop-out-rate 0.1 --lr 5e-5 --reg-strength 0.001 --num-workers 4 \
--use-cosine-scheduler --lr-restart-epoch 1 --lr-restart-mul 2 \
--lr-decay-factor 0.5 --lr-warm-up 0.1 --min-lr 1e-8 \
--temporal-scale 3 --eval-per-epoch 3
```

- MSRVTT-QA
```
python train.py --dataset msrvtt-qa-oe \
--dataset-dir <path/to/dataset> --ckpt-interval 2 --batch-size 10 \
--epoch 7 --drop-out-rate 0.1 --lr 2e-5 --reg-strength 0.001 --num-workers 4 \
--use-cosine-scheduler --lr-restart-epoch 1 --lr-restart-mul 2 \
--lr-decay-factor 1 --lr-warm-up 0.05 --min-lr 1e-8 \
--temporal-scale 3 --eval-per-epoch 3
```

- TGIF-FrameQA
```
python train.py --dataset tgif-frameqa \
--dataset-dir <path/to/dataset> --ckpt-interval 3 --batch-size 10 \
--epoch 15 --drop-out-rate 0.1 --lr 1e-4 --reg-strength 0.001 --num-workers 4 \
--use-cosine-scheduler --lr-restart-epoch 1 --lr-restart-mul 2 \
--lr-decay-factor 0.5 --lr-warm-up 0.1 --min-lr 1e-8 \
--temporal-scale 3 --eval-per-epoch 3
```

- TGIF-Transition
```
python train.py --dataset tgif-transition \
--dataset-dir <path/to/dataset> --ckpt-interval 3 --batch-size 9 \
--epoch 5 --drop-out-rate 0.1 --lr 2e-5 --reg-strength 0.001 --num-workers 4 \
--use-cosine-scheduler --lr-restart-epoch 1 --lr-restart-mul 2 \
--lr-decay-factor 1 --lr-warm-up 0 --min-lr 1e-8 \
--temporal-scale 3 --eval-per-epoch 3
```

- TGIF-Action
```
python train.py --dataset tgif-action \
--dataset-dir <path/to/dataset> --ckpt-interval 3 --batch-size 16 \
--epoch 10 --drop-out-rate 0.1 --lr 3e-5 --reg-strength 0.001 --num-workers 4 \
--use-cosine-scheduler --lr-restart-epoch 1 --lr-restart-mul 2 \
--lr-decay-factor 1 --lr-warm-up 0.1 --min-lr 1e-8 \
--temporal-scale 3 --eval-per-epoch 3
```

<b>Note:</b> We trained our models with 4 GPUs and utilize the ddp training strategy, so the results might vary when the models is trained under different number of GPUs due to the batch size. 

## Evaluation
To perform evaluation on a trained model, you can type in the following:
```
python eval.py --dataset <dataset/name> \
--dataset-dir <path/to/dataset> \
--batch-size 32 --num-workers 4 --temporal-scale 3 \
--model-path <path/to/model>
```
The `dataset` arguments can be:
- `msvd-qa-oe` for MSVD-QA
- `msvrvtt-qa-oe` for MSVD-QA
- `tgif-frameqa` for TGIF-FrameQA
- `tgif-transition` for TGIF-Transition
- `tgif-action` for TGIF-Action

To get our reported performance, download our best training checkpoints <a href="https://sejonguniversity-my.sharepoint.com/:f:/g/personal/22110338_sju_ac_kr/Ej-phn6QYMRFqdATVOEMt5ABj3pxNlaIGq3d63I76dhwcg?e=WtonoL">here</a>.

## Citation
```
@article{Immanuel2023,
    author  = {S. A. Immanuel and C. Jeong},
    title   = {Lightweight recurrent cross-modal encoder for video question answering},
    journal = {Knowledge-Based Systems},
    volume  = {},
    number  = {},
    pages   = {},
    month   = {6},
    year    = {2023},
}
```

## Credits
- https://github.com/xudejing/video-question-answering
- https://github.com/YunseokJANG/tgif-qa
- https://github.com/SwinTransformer/Video-Swin-Transformer


