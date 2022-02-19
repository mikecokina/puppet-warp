import numpy as np


class _DefaultSettings(object):
    FLOAT_DTYPE = np.float32
    INDEX_DTYPE = np.uint32


class Settings(_DefaultSettings):
    _instance = None

    # defaults #########################################################################################################
    DEFAULT_SETTINGS = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls.DEFAULT_SETTINGS = cls.settings_serializer()
            # Put any initialization here.
            cls.read_and_update_config()

        return cls._instance

    @classmethod
    def settings_serializer(cls):
        return {
            "DTYPE": cls.FLOAT_DTYPE,
        }

    @classmethod
    def read_and_update_config(cls, conf_path=None):
        cls.update_config()

    @classmethod
    def configure(cls, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(cls, key):
                raise ValueError("You are about to set configuration which doesn't exist")
            setattr(cls, key, value)

    @classmethod
    def update_config(cls):
        cls.CUDA = False


settings = Settings()
