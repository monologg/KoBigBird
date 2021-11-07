from evaluate.cls import eval_cls
from evaluate.qa import eval_qa

EVAL_FUNC_MAP = {"cls": eval_cls, "qa": eval_qa}
