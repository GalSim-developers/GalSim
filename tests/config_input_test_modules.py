import galsim.config
import galsim.config.value_eval

galsim.config.value_eval.eval_base_variables += ["input_size"]


def _ret_size(*, size):
    return size


class InputSizeLoader(galsim.config.InputLoader):
    def getKwargs(self, config, base, logger):
        req = {"size": int}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
        if self.takes_logger:
            kwargs["logger"] = logger
        return kwargs, True

    def initialize(self, input_objs, num, base, logger):
        base['input_size'] = input_objs[0]


galsim.config.RegisterInputType(
    "input_size_module",
    InputSizeLoader(_ret_size, file_scope=True),
)
