import galsim.config
import galsim.config.value_eval
import numpy as np

galsim.config.eval_base_variables += ["input_size_0", "input_size_arr"]


def _ret_size(size=-1):
    return size


class InputSizeLoader(galsim.config.InputLoader):
    def getKwargs(self, config, base, logger):
        req = {"size": int}
        kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
        if self.takes_logger:
            kwargs["logger"] = logger
        return kwargs, True

    def initialize(self, input_objs, num, base, logger):
        if num == 0:
            base['input_size_0'] = input_objs[0]
        if all(iobj is not None for iobj in input_objs):
            base['input_size_arr'] = np.array(input_objs, dtype=int)


galsim.config.RegisterInputType(
    "input_size_module",
    InputSizeLoader(_ret_size, file_scope=True),
)
