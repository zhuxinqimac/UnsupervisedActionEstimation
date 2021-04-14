from models.beta import BetaCeleb, BetaShapes
from models.forward_vae import forward_vae, beta_forward
from models.group_vae import rl_group_vae, forward_grvae
from models.factor_vae import factor_vae, factor_conv_vae
from models.dip_vae import dip_vae, dip_conv_vae
from models.lie_vae import LieCeleb
from models.lie_vae_action import LieAction
from models.lie_vae_action_simple import LieActionSimple
from models.lie_vae_rl import lie_rl_group_vae
from models.uneven_vae import UnevenVAE
from models.uneven_facvae import UnevenFacVAE
from models.uneven_dipvae import UnevenDIPVAE

models = {
    'beta_shapes': BetaShapes,
    'beta_celeb': BetaCeleb,
    'forward': forward_vae,
    'rgrvae': rl_group_vae(False),
    'factor_vae': factor_vae,
    'factor_conv_vae': factor_conv_vae,
    'dip_vae_i': dip_vae,
    'dip_vae_ii': dip_vae,
    'dip_conv_vae_i': dip_conv_vae,
    'dip_conv_vae_ii': dip_conv_vae,
    'beta_forward': beta_forward,
    'dforward': forward_grvae(False),
    'lie_group': LieCeleb,
    'lie_group_action': LieAction,
    'lie_group_action_simple': LieActionSimple,
    'lie_group_rl': lie_rl_group_vae(False),
    'uneven_vae': UnevenVAE,
    'uneven_facvae': UnevenFacVAE,
    'uneven_dip_vae_i': UnevenDIPVAE,
    'uneven_dip_vae_ii': UnevenDIPVAE,
}
