import galsim

# This extra output type simply saves the values of the image offsets when an 
# object is drawn into the stamp. This can then be used by e.g. the MEDSBuilder
from .extra import ExtraOutputBuilder
class OffsetBuilder(ExtraOutputBuilder):
    """This saves the stamp offset values for later use"""
    # The function to call at the end of building each stamp
    def processStamp(self, obj_num, config, base, logger):
        offset = base['stamp_offset']
        stamp = base['stamp']
        if 'offset' in stamp:
            offset += galsim.config.GetCurrentValue('offset', base['stamp'],
                                                            galsim.PositionD, base)
        self.scratch[obj_num] = offset

    # The function to call at the end of building each file to finalize the truth catalog
    def finalize(self, config, base, main_data, logger):
        offsets_list = []
        obj_nums = sorted(self.scratch.keys())
        for obj_num in obj_nums:
            offsets_list.append(self.scratch[obj_num])
        return offsets_list

# Register this as a valid extra output
from .extra import RegisterExtraOutput
RegisterExtraOutput('offset', OffsetBuilder())
