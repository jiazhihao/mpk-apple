"""Automatic fusion boundaries validated on the 40-core M5 Max."""


def finalize(program, graph, emitted, config, *, t, dynamic_t, speculative,
             commute_norm, accelerator, gdn_mixer_fusion, barriers):
    if (gdn_mixer_fusion and config.gdn_mixer_fusion and config.key == "apple10"
            and config.gpu_cores == 40 and t == 8 and not dynamic_t and not speculative
            and commute_norm and accelerator == "on" and config.sibling_order != "bus_first"):
        from ....compiler.gdn_fusion import apply_gdn_mixer_fusion
        from ....compiler.barriers import place_barriers
        program = apply_gdn_mixer_fusion(program, graph, emitted, config.gdn_mixer_fusion)
        place_barriers(program, barriers)
    return program
