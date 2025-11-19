import importlib
VOCODERS = {}


def register_vocoder(cls):
    VOCODERS[cls.__name__.lower()] = cls
    VOCODERS[cls.__name__] = cls
    return cls


def get_vocoder_cls(hparams):
    name = (hparams.get('vocoder') or 'pwg')
    if name in VOCODERS:
        return VOCODERS[name]
    else:
        # 支援 "pkg.ClassName" 形式的動態載入
        if '.' in name:
            pkg = ".".join(name.split(".")[:-1])
            cls_name = name.split(".")[-1]
            return getattr(importlib.import_module(pkg), cls_name)
        raise ValueError(f"Unknown vocoder: {name}")

    # if hparams['vocoder'] in VOCODERS:
    #     return VOCODERS[hparams['vocoder']]
    # else:
    #     vocoder_cls = hparams['vocoder']
    #     pkg = ".".join(vocoder_cls.split(".")[:-1])
    #     cls_name = vocoder_cls.split(".")[-1]
    #     vocoder_cls = getattr(importlib.import_module(pkg), cls_name)
    #     return vocoder_cls


class BaseVocoder:
    def spec2wav(self, mel):
        """

        :param mel: [T, 80]
        :return: wav: [T']
        """

        raise NotImplementedError

    @staticmethod
    def wav2spec(wav_fn):
        """

        :param wav_fn: str
        :return: wav, mel: [T, 80]
        """
        raise NotImplementedError
