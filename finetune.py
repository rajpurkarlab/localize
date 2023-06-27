from .lseg.modules.lseg_module import LSegModule
from .lseg.utils import get_default_argument_parser
from model import Model

parser = LSegModule.add_model_specific_args(get_default_argument_parser())
hparams = parser.parse_args()
lseg = LSegModule(**vars(hparams)) # Load demo weights

model = Model(lseg)
