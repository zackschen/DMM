from pprint import pprint
import argparse

def parse_opt():

    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--dataset', type=str, default='refcoco', help='name of dataset')
    parser.add_argument('--splitBy', type=str, default='unc', help='who splits this dataset')
    parser.add_argument('--start_from', type=str, default=None, help='continuing training from saved model')
    # FRCN setting
    parser.add_argument('--imdb_name', default='coco_minus_refer', help='image databased trained on.')
    parser.add_argument('--net_name', default='res101', help='net_name: res101 or vgg16')
    parser.add_argument('--iters', default=1250000, type=int, help='iterations we trained for faster R-CNN')
    parser.add_argument('--tag', default='notime', help='on default tf, don\'t change this!')
    # Visual Encoder Setting
    parser.add_argument('--visual_sample_ratio', type=float, default=0.3, help='ratio of same-type objects over different-type objects')
    parser.add_argument('--visual_fuse_mode', type=str, default='concat', help='concat or mul')
    parser.add_argument('--visual_init_norm', type=float, default=20, help='norm of each visual representation')
    parser.add_argument('--visual_use_bn', type=int, default=-1, help='>0: use bn, -1: do not use bn in visual layer')
    parser.add_argument('--visual_use_cxt', type=int, default=1, help='if we use contxt')
    parser.add_argument('--visual_cxt_type', type=str, default='frcn', help='frcn or res101')
    parser.add_argument('--visual_drop_out', type=float, default=0.2, help='dropout on visual encoder')
    parser.add_argument('--window_scale', type=float, default=2.5, help='visual context type')
    # Visual Feats Setting
    parser.add_argument('--with_st', type=int, default=1, help='if incorporating same-type objects as contexts')
    parser.add_argument('--num_cxt', type=int, default=5, help='how many surrounding objects do we use')
    # Language Encoder Setting
    parser.add_argument('--word_embedding_size', type=int, default=512, help='the encoding size of each token')
    parser.add_argument('--word_vec_size', type=int, default=512, help='further non-linear of word embedding')
    parser.add_argument('--word_drop_out', type=float, default=0.5, help='word drop out after embedding')
    parser.add_argument('--bidirectional', type=int, default=1, help='bi-rnn')
    parser.add_argument('--rnn_hidden_size', type=int, default=512, help='hidden size of LSTM')
    parser.add_argument('--rnn_type', type=str, default='lstm', help='rnn, gru or lstm')
    parser.add_argument('--rnn_drop_out', type=float, default=0.2, help='dropout between stacked rnn layers')
    parser.add_argument('--rnn_num_layers', type=int, default=1, help='number of layers in lang_encoder')
    parser.add_argument('--variable_lengths', type=int, default=1, help='use variable length to encode')
    # Joint Embedding setting
    parser.add_argument('--jemb_drop_out', type=float, default=0.1, help='dropout in the joint embedding')
    parser.add_argument('--jemb_dim', type=int, default=512, help='joint embedding layer dimension')
    # Common space setting
    parser.add_argument('--cemb_dim', type=int, default=1024, help='joint embedding layer dimension')
    parser.add_argument('--loc_dim', type=int ,default=5, help='location embdedding dimension')
    # Loss Setting
    parser.add_argument('--att_weight', type=float, default=1.0, help='weight on attribute prediction')
    parser.add_argument('--visual_rank_weight', type=float, default=1.0, help='weight on paired (ref, sent) over unpaired (neg_ref, sent)')
    parser.add_argument('--lang_rank_weight', type=float, default=1.0, help='weight on paired (ref, sent) over unpaired (ref, neg_sent)')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for ranking loss')
    parser.add_argument('--reg_lambda', type=float, default=1.0, help='weight on box regression')
    # Optimization: General
    parser.add_argument('--max_iters', type=int, default=50000, help='max number of iterations to run')
    parser.add_argument('--max_category_iters', type=int, default=10000, help='max number of iterations of one category to run')
    parser.add_argument('--max_category_epoch', type=int, default=15, help='max number of epoch of one category to run')
    parser.add_argument('--sample_ratio', type=float, default=0.3, help='ratio of same-type objects over different-type objects')
    parser.add_argument('--batch_size', type=int, default=15, help='batch size in number of images per batch')
    parser.add_argument('--ref_batch_size', type=int, default=45, help='batch size in number of refs per batch')
    parser.add_argument('--cross_batch_size', type=int, default=39, help='batch size in number of refs per cross batch')
    parser.add_argument('--num_props', type=int, default=100, help='number of proposals per object')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value')
    parser.add_argument('--seq_per_ref', type=int, default=3, help='number of expressions per object during training')
    parser.add_argument('--cross_seq_per_ref', type=int, default=1, help='number of expressions per object during training')
    parser.add_argument('--learning_rate_decay_start', type=int, default=8000, help='at what iter to start decaying learning rate')
    parser.add_argument('--learning_rate_decay_every', type=int, default=8000, help='every how many iters thereafter to drop LR by half')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--learning_rate', type=float, default=4e-4, help='learning rate')
    parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
    # Evaluation/Checkpointing
    parser.add_argument('--num_sents', type=int, default=-1, help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=600, help='how often to save a model checkpoint?')
    parser.add_argument('--checkpoint_path', type=str, default='output', help='directory to save models')
    parser.add_argument('--log_path', type=str, default='./log', help='directory to save log')
    parser.add_argument('--language_eval', type=int, default=0, help='Evaluate language as well (1 = yes, 0 = no)?')
    parser.add_argument('--losses_log_every', type=int, default=25, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1, help='Do we load previous best score when resuming training.')
    # misc
    parser.add_argument('--id', type=str, default='0', help='an id identifying this run/job.')
    parser.add_argument('--seed', type=int, default=24, help='random number generator seed to use')
    parser.add_argument('--gpuid', type=int, default=0, help='which gpu to use, -1 = use CPU')
    # sliding window
    parser.add_argument('--k_eps', type=float, default=1.25e-6, help='flag loss have converged')
    parser.add_argument('--window_size', type=int, default=500, help='sliding window size')
    # mas
    parser.add_argument('--mas_weighted_decay', type=bool, default=False, help='flag for using weighted_decay or not')
    parser.add_argument('--weighted', type=int, default=0, help='shared or weighted type')
    parser.add_argument('--f_id', type=str, default='0', help='an id identifying the first model of a sequence task.')
    parser.add_argument('--freeze_id', type=int, default=0, help='an id switching the case of different freeze_layers.')
    parser.add_argument('--module_sum', type=float, default=0.0, help='flag for using module sum or mean')
    parser.add_argument('--module_normalize', type=float, default=0.0, help='flag for using normalize')
    parser.add_argument('--sub_module', type=float, default=0.0, help='flag for subject using module weighted')
    # replay buffer
    parser.add_argument('--buffer_type', type=str, default="low", help='replay buffer type')
    parser.add_argument('--buffer_size', type=float, default=120, help='buffer size')
    parser.add_argument('--buffer_sample_number', type=float, default=100, help='number of sample in buffer(0~100) to train')
    parser.add_argument('--buffer_lambda', type=float, default=1.0, help='weight on replay buffer')
    parser.add_argument('--buffer_start_epoch', type=int, default=15,
                        help='the epoch of begining to update current buffer')
    parser.add_argument('--subject_flag', type=float, default=0.0, help='flag for using subject buffer')
    parser.add_argument('--multi_buffer', type=float, default=0.0, help='flag for using three buffer')

    parser.add_argument('--gdumb_size', type=int, default=5000, help='memory size for gdumb')
    parser.add_argument('--task', type=int, default=5, help='number of tasks')
    
    # parse
    args = parser.parse_args()
    opt = vars(args)
    pprint('parsed input parameters:')
    pprint(opt)
    return args

if __name__ == '__main__':

    opt = parse_opt()
    print('opt[\'id\'] is ', opt['id'])




