class AbstractArchitecture(object):
    def base(base, pretrained):
        raise NotImplemented

    def forward(base, misc):
        raise NotImplemented