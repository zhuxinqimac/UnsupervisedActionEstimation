import argparse
import ast
from trainer import run
from utils import _str_to_list_of_int, _str_to_list_of_str, _str_to_bool

parser = argparse.ArgumentParser('Disentanglement')
# Basic Training Args
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--model', default='forward', type=str, choices=['beta_shapes', 'beta_celeb', 'forward', 'rgrvae', 'dip_vae_i', 'dip_vae_ii',
                                                                     'beta_forward', 'dforward', 'lie_group', 'lie_group_action', 'lie_group_action_simple',
                                                                     'lie_group_rl', 'factor_vae', 'factor_conv_vae', 'dip_conv_vae_i', 'dip_conv_vae_ii',
                                                                     'uneven_vae', 'uneven_facvae', 'uneven_dip_vae_i', 'uneven_dip_vae_ii',
                                                                     'diffdim_vae_64', 'dimvar_vae_64'])
parser.add_argument('--dataset', default='flatland', type=str, choices=['flatland', 'dsprites', 'teapot', 'teapot_nocolor', 'shapes3d', 'stylegan3d', 'stylegandsp'])
parser.add_argument('--fixed_shape', default=None, type=int, help='Fixed shape in dsprites. None for not fixed.')
parser.add_argument('--data-path', default=None, type=str, help='Path to dataset root')
parser.add_argument('--latents', default=4, type=int, help='Number of latents')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--split', default=0.1, type=float, help='Validation split fraction')
parser.add_argument('--shuffle', default=True, type=ast.literal_eval, help='Shuffle dataset')
parser.add_argument('--lr-scheduler', default='none', choices=['exp', 'none'], type=str)
parser.add_argument('--lr-scheduler-gamma', default=0.99, type=float, help='Exponential lr scheduler gamma')

# Model Loading
parser.add_argument('--base-model', default='beta_forward', type=str, help='Base model for rgrvae, dforward')
parser.add_argument('--base-model-path', default=None, help='Path to base model state which is to be loaded')
parser.add_argument('--load-model', default=False, type=ast.literal_eval, help='Continue training by loading model')
parser.add_argument('--log-path', default=None, type=str, help='Path from which to load model')
parser.add_argument('--global-step', default=None, help='Set the initial logging step value', type=int)

# Learning Rates
parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--group-learning-rate', default=0.1, type=float, help='Learning rate for internal rgrvae matrix representations')
parser.add_argument('--policy-learning-rate', default=None, type=float, help='Learning rate for policy network')

# Hyperparams
parser.add_argument('--beta', default=1., type=float, help='Beta vae beta')
parser.add_argument('--capacity', default=None, type=float, help='KL Capacity')
parser.add_argument('--capacity-leadin', default=100000, type=int, help='KL capacity leadin')
parser.add_argument('--gvae-gamma', default=None, type=float, help='Weighting for prediction loss. =beta if None')

# Metrics And Vis
parser.add_argument('--visualise', default=True, type=ast.literal_eval, help='Do visualisations')
parser.add_argument('--metrics', default=False, type=ast.literal_eval, help='Calculate disentanglement metrics at each step')
parser.add_argument('--end-metrics', default=True, type=ast.literal_eval, help='Calculate disentanglement metrics at end of run')
parser.add_argument('--evaluate', default=False, type=ast.literal_eval, help='Only run evalulation')

# RGrVAE Args
parser.add_argument('--offset', default=1, type=int, help='Generative factor offset for each action')
parser.add_argument('--use-regret', default=False, type=ast.literal_eval, help='Use regret on reinforce models')
parser.add_argument('--base-policy-weight', default=1., type=float, help='Base weight at which to apply policy over random choice')
parser.add_argument('--base-policy-epsilon', default=0.0000, type=float)
parser.add_argument('--group-structure', default=['c+', 'c-'], type=str, nargs='+', help='Group structure per latent pair in group vae. Options in models/actions: GroupWiseAction')
parser.add_argument('--num-action-steps', default=1, type=int, help='Number of action steps to allow Reinforced GroupVAE')
parser.add_argument('--multi-action-strategy', default='reward', choices=['reward', 'returns'], help='Strategy for reinforcing multiple actions.')
parser.add_argument('--reinforce-discount', default=0.99, type=float, help='Discount factor for reinforce rewards/returns')
parser.add_argument('--use-cob', default=False, type=ast.literal_eval, help='Use change of basis for representaitons in gvae')
parser.add_argument('--entropy-weight', default=0.01, type=float, help='Entropy weight for RL exploration')
parser.add_argument('--noise-name', default=None, choices=['BG', 'Salt', 'Gaussian'])

# ForwardVAE Params
parser.add_argument('--pred-z-loss-type', default='latent', type=str, help='Pred loss in ForwardVAE.')

# Lie Group Model
parser.add_argument('--subgroup_sizes_ls', default=[25, 25, 25, 25], type=_str_to_list_of_int, help='Subgroup sizes list for subspace group vae')
parser.add_argument('--subspace_sizes_ls', default=[1, 1, 1, 1], type=_str_to_list_of_int, help='Subspace sizes list for subspace group vae')
parser.add_argument('--lie_alg_init_type_ls', default=['none', 'none', 'none', 'none'], type=_str_to_list_of_str, help='Lie_alg init type list for subspace group vae')
parser.add_argument('--hy_ncut', default=0, type=int, help='Hyper-param for number of cuts in LieGroupVAE.')
parser.add_argument('--normalize_alg', default=False, type=_str_to_bool, help='If normalize alg in norm_vae')
parser.add_argument('--use_alg_var', default=False, type=_str_to_bool, help='If use alg_var in norm_vae')
parser.add_argument('--lie_alg_init_scale', help='Hyper-param for lie_alg_init_scale.', default=0.001, type=float)
parser.add_argument('--hy_hes', default=0, type=float, help='Hyper-param for Hessian in LieVAE.')
parser.add_argument('--hy_rec', default=1, type=float, help='Hyper-param for gfeats reconstruction in GroupVAE and LieVAE.')
parser.add_argument('--hy_commute', default=0, type=float, help='Hyper-param for commutator in GroupVAE-v3.')
parser.add_argument('--cycle_latent', default=False, type=_str_to_bool, help='If use cycle_latent in LieGroupVAE.')
parser.add_argument('--cycle_limit', default=4., type=float, help='The limit of cycle_latent in LieGroupVAE.')
parser.add_argument('--cycle_prob', default=0.1, type=float, help='The limit of cycle_latent in LieGroupVAE.')
parser.add_argument('--forward_eg_prob', default=0.6667, type=float, help='The prob to forward eg in LieGroupVAE.')
parser.add_argument('--recons_loss_type', default='l2', choices=['l2', 'bce'], type=str, help='The reconstruction type for x.')
parser.add_argument('--no_exp', default=False, type=_str_to_bool, help='If deactivate exp_mapping in LieGroupVAE.')

# Lie Group Action Model
parser.add_argument('--num_actions', default=4, type=int, help='Hyper-param for number of actions in LieAction.')
parser.add_argument('--loss_type', default='on_group', type=str, help='Loss type.')

# Lie Group RL Model
parser.add_argument('--supervised_train', default=False, type=_str_to_bool, help='If use action supervision in LieGroupVAERL.')
parser.add_argument('--hy_st', default=0., type=float, help='Hyper-param for action strength LieGroupVAERL.')

# Uneven Model
parser.add_argument('--uneven_reg_maxval', default=0, type=float, help='The maxval of the lin_reg in uneven vae loss.')
# parser.add_argument('--exp_uneven_reg', default=False, type=_str_to_bool, help='If use exp reg rather than lin reg in uneven vae loss.')
parser.add_argument('--uneven_reg_lambda', default=0, type=float, help='The (decoder) lambda for reg in uneven vae loss.')
parser.add_argument('--uneven_reg_encoder_lambda', default=0, type=float, help='The encoder lambda for reg in uneven vae loss.')
parser.add_argument('--uneven_masked_w', default=False, type=_str_to_bool, help='If use masked W in first layer of uneven vae decoder.')
# parser.add_argument('--use_cumax_adaptive', default=False, type=_str_to_bool, help='If use adaptive reg with cumax in uneven vae loss.')
parser.add_argument('--reg_type', default='linear', choices=['linear', 'exp', 'cumax_ada', 'monoconst_ada'],
                    type=str, help='What type of regularization to use in uneven vae loss.')
parser.add_argument('--orth_lambda', default=0, type=float, help='The orthogonal reg lambda in uneven vae.')
parser.add_argument('--uneven_w_max_dropout_rate', default=0, type=float, help='The maximal dropout rate in w in uneven vae decoder (between [0, 1)).')
parser.add_argument('--xav_init', default=False, type=_str_to_bool, help='If use Xavier init.')
parser.add_argument('--lpips_lambda', default=0, type=float, help='The lpips lambda in uneven vae.')

# FactorVAE
parser.add_argument('--factor_vae_gamma', default=6.4, type=float, help='The gamma in factor vae.')

# DIP-VAE
parser.add_argument('--lambda_d', default=1, type=float, help='The lambda_d in DIP vae.')
parser.add_argument('--lambda_od', default=10, type=float, help='The lambda_od in DIP vae.')

# DiffDim-VAE
parser.add_argument('--lpips_net', default='alex', type=str, help='The lpips net type.')
parser.add_argument('--var_sample_scale', default=0.05, type=float, help='The var scale in diffdim loss.')
parser.add_argument('--norm_on_depth', default=False, type=_str_to_bool, help='If normalize on depth in diffdim loss.')
parser.add_argument('--sensor_used_layers', default=5, type=int, help='The number of used lpips layers in diffdim loss.')
parser.add_argument('--norm_lambda', default=0., type=float, help='The norm lambda in diffdim loss.')
parser.add_argument('--use_norm_mask', default=True, type=_str_to_bool, help='If use norm mask in cos_fn in diffdim loss.')
parser.add_argument('--divide_mask_sum', default=True, type=_str_to_bool, help='If divide the maks_sum in cos_fn in diffdim loss.')
parser.add_argument('--diff_lambda', default=1, type=float, help='The main lambda of diffdim loss.')
parser.add_argument('--diff_capacity_leadin', default=None, type=int, help='Diff loss capacity leadin')
parser.add_argument('--diff_capacity', default=None, type=float, help='Diff Capacity')
parser.add_argument('--train_lpips', default=False, type=_str_to_bool, help='If train lpips in diffdim loss.')
parser.add_argument('--use_dynamic_scale', default=False, type=_str_to_bool, help='If use dynamic scale in diffdim loss.')
parser.add_argument('--detach_qpn', default=False, type=_str_to_bool, help='If detach q, pos and neg in diffdim loss.')
parser.add_argument('--aux_opt_type', default='decoder_first', type=str, help='The auxiliary optimizer type.')
parser.add_argument('--use_norm_as_mask', default=False, type=_str_to_bool, help='If use norm as mask.')
parser.add_argument('--diff_avg_lerp_rate', default=0.01, type=float, help='The lerp rate for diff_avg.')
parser.add_argument('--lerp_lambda', default=0., type=float, help='Lerp lambda.')
parser.add_argument('--neg_lambda', default=1., type=float, help='negative samples lambda.')
parser.add_argument('--train_with_vae', default=False, type=_str_to_bool, help='If train diffdim with vae.')

# DimVar-VAE
parser.add_argument('--eval_dataset', default=None, type=str, choices=['flatland', 'dsprites', 'teapot', 'teapot_nocolor', 'shapes3d'])
parser.add_argument('--eval_data_path', default=None, type=str)

args = parser.parse_args()


if __name__ == '__main__':
    from models.utils import ContextTimer
    with ContextTimer('Run'):
        run(args)
