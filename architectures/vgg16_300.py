import torch
from torchvision.models import vgg16 as VGG16_model
from .abstract_architecture import AbstractArchitecture
class Arch300_VGG16(AbstractArchitecture):
    NAME = '300_VGG16'
    """
        The basis of this architecture is vgg16.
        It will be fully taken until the Conv5_3 layer (num 29)
        Then the following layers corresponding to the extras will be added:
        - Conv6 (replaces fc6).
        - Conv7 (replaces fc7).
        - Conv8_1
        - Conv8_2
        - Conv9_1
        - Conv9_2
        - Conv10_1
        - Conv10_2
        - Conv11_1
        - Conv11_2

        For the location layers and confidences, apply a convolution to the following layers:
        - Conv4_3 applying a l2norm to avoid very high values.
        - Conv7
        - Conv8_2
        - Conv9_2
        - Conv10_2
        - Conv11_2

        Change the Maxpool4_3 to ceil_mode = True

        Each selected layer has n associated priors:
        - Conv4_3: has 4 and its feature map is 38.
        - Conv7: has 6 and its feature map is 19.
        - Conv8_2: has 6 and its feature map is 10.
        - Conv9_2: has 6 and its feature map is 5.
        - Conv10_2: has 4 and its feature map is 3.
        - Conv11_2: has 4 and its feature map is 1.

        The total is 38 * 38 * 4 + 19 * 19 * 6 + 10 * 10 * 6 + 5 * 5 * 6 + 3 * 3 * 4 + 1 * 1 * 4 = 8732  possible detections in differents scales, sizes and locations.
    """
    class L2Norm(torch.nn.Module):
        def __init__(self, n_channels, scale):
            super(Arch300_VGG16.L2Norm, self).__init__()
            self.n_channels = n_channels
            self.gamma = scale or None
            self.eps = 1e-10
            self.weight = torch.nn.Parameter(torch.Tensor(self.n_channels))
            self.reset_parameters()

        def reset_parameters(self):
            torch.nn.init.constant_(self.weight, self.gamma)

        def forward(self, x):
            norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
            #x /= norm
            x = torch.div(x, norm)
            out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
            return out

    image_size = 300

    normalization = {
        'mu':[104, 117, 123],
        'sigma': [1, 1, 1]
    }

    def base(pretrained):
        out_channels_of_layers_with_classification = []
        CONV5_3 = 29 
        OUT_CHANNELS_CONV5_3 = 512
        MAXPOOL_4_3 = 16
        
        # Get base
        base = VGG16_model(pretrained=pretrained)
            
        # Select only the first n layers.
        base = base.features[:1 + CONV5_3]
        
        # Apply the changes
        base[MAXPOOL_4_3].ceil_mode = True
        
        # Add in out_channels_of_layers_with_classification the layers in base that have confidences and locations
        OUT_CHANNELS_CONV4_3 = 512
        out_channels_of_layers_with_classification.append(OUT_CHANNELS_CONV4_3)

        return base, {'l2norm_conv4_3': Arch300_VGG16.L2Norm(n_channels=OUT_CHANNELS_CONV4_3, scale=20)}, OUT_CHANNELS_CONV5_3, out_channels_of_layers_with_classification

    def forward(base, x, other_layers):
        CONV4_3 = 22
        results_with_classification = []
        for i, v in enumerate(base):
            x = v(x)
            if i == CONV4_3:
                results_with_classification.append(other_layers['l2norm_conv4_3'](x)) # ['l2norm_conv4_3']
        return x, results_with_classification

    extras = [
        ('M', {'kernel_size': 3, 'stride': 1, 'padding': 1, 'ceil_mode': True}), # MaxPool
        
        ('C', {'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 6, 'dilation': 6}), # Conv6
        
        ('Cc', {'out_channels': 1024, 'kernel_size': 1}), # Conv7
        
        # Extras
        ('C', {'out_channels': 256, 'kernel_size': 1, 'stride': 1}), # Conv8_1
        ('Cc', {'out_channels': 512, 'kernel_size': 3, 'stride': 2, 'padding': 1}), # Conv8_2

        ('C', {'out_channels': 128, 'kernel_size': 1, 'stride': 1}), # Conv9_1
        ('Cc', {'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1}), # Conv9_2

        ('C', {'out_channels': 128, 'kernel_size': 1, 'stride': 1}), # Conv10_1
        ('Cc', {'out_channels': 256, 'kernel_size': 3, 'stride': 1}), # Conv10_2

        ('C', {'out_channels': 128, 'kernel_size': 1, 'stride': 1}), # Conv11_1
        ('Cc', {'out_channels': 256, 'kernel_size': 3, 'stride': 1}), # Conv11_2
    ]

    priors = {
        'clip': True,
        'configs': lambda define: [
            define(num_dim=38, step=8,   size=(30, 60),   aspect_ratios=[2]), # Conv4_3
            define(num_dim=19, step=16,  size=(60, 111),  aspect_ratios=[2, 3]), # Conv7
            define(num_dim=10, step=32,  size=(111, 162), aspect_ratios=[2, 3]), # Conv8_2
            define(num_dim=5,  step=64,  size=(162, 213), aspect_ratios=[2, 3]), # Conv9_2
            define(num_dim=3,  step=100, size=(213, 264), aspect_ratios=[2]), # Conv10_2
            define(num_dim=1,  step=300, size=(264, 315), aspect_ratios=[2]), # Conv11_2
        ]
    }