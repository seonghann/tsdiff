def get_model(config):
    if config.network == "dualenc":
        from .dualenc import DualEncoderEpsNetwork
        return DualEncoderEpsNetwork(config)

    elif config.network == "dualenc_general":
        from .dualenc_dimepp_newedge_nolocal import DualEncoderEpsNetwork
        return DualEncoderEpsNetwork(config)

    elif config.network == "condensenc":
        from .condensenc import CondenseEncoderEpsNetwork
        return CondenseEncoderEpsNetwork(config)

    else:
        raise NotImplementedError("Unknown network: %s" % config.network)
