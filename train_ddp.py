import parser
import sys

from lrce.lib import *
from lrce.models.e2e import E2EOpenEnded, E2EMultipleChoice, E2ECount
from lrce.dataset.e2e_dataset import E2EMicrosoftDataset, E2ETGIFDataset
from lrce.agent import AgentOE, AgentMC, AgentCount


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group('nccl', rank=rank, world_size=world_size)


def main(rank: int, world_size: int, train_args: argparse.Namespace):
    ddp_setup(rank, world_size)
    T.cuda.set_device(rank)
    T.cuda.empty_cache()

    setup_logging()
    logger = get_logger(__name__, rank)

    logger.info('Preparing dataset')
    if 'tgif' in train_args.dataset:
        tgif_type = train_args.dataset.split('-')[-1]
        train_dataset = E2ETGIFDataset(
            split_annotation=f'{train_args.dataset_dir}/annotations/Train_{tgif_type}_question.csv',
            full_annotation=f'{train_args.dataset_dir}/annotations/Total_{tgif_type}_question.csv',
            videos_path=f'{train_args.dataset_dir}/gifs',
            max_text_token_len=train_args.text_seq_len,
            task_type=train_args.task_type,
            sanity_check=train_args.sanity_check,
            frames_per_clip=train_args.frame_sample_size,
            temporal_scale=train_args.temporal_scale,
        )
        val_dataset = E2ETGIFDataset(
            split_annotation=f'{train_args.dataset_dir}/annotations/Test_{tgif_type}_question.csv',
            full_annotation=f'{train_args.dataset_dir}/annotations/Total_{tgif_type}_question.csv',
            videos_path=f'{train_args.dataset_dir}/gifs',
            max_text_token_len=train_args.text_seq_len,
            task_type=train_args.task_type,
            sanity_check=train_args.sanity_check,
            frames_per_clip=train_args.frame_sample_size,
            temporal_scale=train_args.temporal_scale,
        )
    else:
        video_dict = pickle.load(open(os.path.join(train_args.dataset_dir, 'idx-video-mapping.pkl'), 'rb'))

        train_dataset = E2EMicrosoftDataset(
            train_annotation=f'{train_args.dataset_dir}/train_qa.json',
            val_annotation=f'{train_args.dataset_dir}/val_qa.json',
            test_annotation=f'{train_args.dataset_dir}/test_qa.json',
            videos_path=f'{train_args.dataset_dir}/video',
            video_dict=video_dict,
            max_text_token_len=train_args.text_seq_len,
            split='train',
            sanity_check=train_args.sanity_check,
            is_frame_extracted=False,
            temporal_scale=train_args.temporal_scale,
        )
        val_dataset = E2EMicrosoftDataset(
            train_annotation=f'{train_args.dataset_dir}/train_qa.json',
            val_annotation=f'{train_args.dataset_dir}/val_qa.json',
            test_annotation=f'{train_args.dataset_dir}/test_qa.json',
            videos_path=f'{train_args.dataset_dir}/video',
            video_dict=video_dict,
            max_text_token_len=train_args.text_seq_len,
            split='test',
            sanity_check=train_args.sanity_check,
            is_frame_extracted=False,
            temporal_scale=train_args.temporal_scale,
        )

    logger.info('Instantiating model and trainer agent')
    if train_args.task_type == 'oe':
        model_factory = E2EOpenEnded
        agent_factory = AgentOE
    elif train_args.task_type == 'mc':
        model_factory = E2EMultipleChoice
        agent_factory = AgentMC
    elif train_args.task_type == 'count':
        model_factory = E2ECount
        agent_factory = AgentCount
    else:
        logger.error('Unsupported task type')
        sys.exit(-1)

    model = model_factory(
        train_args.feature_dim,
        train_args.num_classes,
        train_args.drop_out_rate,
        train_args.video_feature_res,
        train_args.video_feature_dim,
        train_args.frame_sample_size,
        train_args.temporal_scale,
        train_args.text_seq_len,
    )

    logger.info(f'Using {torch.cuda.device_count()} GPU(s)')
    trainer = agent_factory(model, rank, train_args, not train_args.debug_mode and not train_args.sanity_check)

    if train_args.model_path:
        trainer.load_checkpoint(train_args.model_path)

    logger.info('Instantiating dataloader')
    train_dataloader = T.utils.data.DataLoader(
        train_dataset,
        batch_size=train_args.batch_size,
        shuffle=False,
        num_workers=train_args.num_workers,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset),
    )
    val_dataloader = T.utils.data.DataLoader(
        val_dataset,
        batch_size=train_args.batch_size,
        shuffle=False,
        num_workers=train_args.num_workers,
        pin_memory=True,
        sampler=DistributedSampler(val_dataset),
    )

    if train_args.sanity_check:
        logger.info(
            'Performing sanity check, you should see a very small error or very good metric evaluation on the end result'
        )
        trainer.do_sanity_check(train_dataloader)
    else:
        trainer.do_training(train_dataloader, val_dataloader, train_args.eval_per_epoch)

    destroy_process_group()


if __name__ == '__main__':
    train_args = parser.parse_arg_train()
    world_size = torch.cuda.device_count()
    mp.spawn(main, nprocs=world_size, args=(world_size, train_args))
