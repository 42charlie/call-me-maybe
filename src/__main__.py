from src.utils import parse_args
from src.Engine import Engine

args = parse_args()
engine = Engine(args)
print(engine)