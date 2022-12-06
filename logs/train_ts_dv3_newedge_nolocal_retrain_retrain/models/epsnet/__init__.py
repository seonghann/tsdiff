def get_model(config):
    if config.network == "dualenc":
        from .dualenc import DualEncoderEpsNetwork

        return DualEncoderEpsNetwork(config)

    elif config.network == "dualenc_sch_come":
        from .dualenc_sch_come import DualEncoderEpsNetwork

        return DualEncoderEpsNetwork(config)

    elif config.network == "dualenc_ver3_base_global":
        from .dualenc_ver3_base_global import DualEncoderEpsNetwork

        return DualEncoderEpsNetwork(config)

    elif config.network == "dualenc_ver3_base_nolocal":
        from .dualenc_ver3_base_nolocal import DualEncoderEpsNetwork

        return DualEncoderEpsNetwork(config)

    elif config.network == "dualenc_ver3_newedge_global":
        from .dualenc_ver3_newedge_global import DualEncoderEpsNetwork

        return DualEncoderEpsNetwork(config)

    elif config.network == "dualenc_ver3_newedge_nolocal":
        from .dualenc_ver3_newedge_nolocal import DualEncoderEpsNetwork

        return DualEncoderEpsNetwork(config)

    elif config.network == "mixed2denc_ver1":
        from .mixed2denc_ver1 import Mixed2DEpsNetwork

        return Mixed2DEpsNetwork(config)

    elif config.network == "mixed2denc_ver2":
        from .mixed2denc_ver2 import Mixed2DEpsNetwork

        return Mixed2DEpsNetwork(config)
    else:
        raise NotImplementedError("Unknown network: %s" % config.network)
