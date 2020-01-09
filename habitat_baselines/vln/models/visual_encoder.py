import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

class ResNet(nn.Module):
    r"""A Wrapper class to select the models to use for the visual encoding

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """
    def __init__(self, observation_space):
        super().__init__()
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            self.cnn = models.resnet152(pretrained=True)

        self.layer_init()

    def layer_init(self):
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations):
        cnn_input = []

        # Normalization of the images
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = preprocess(rgb_observations)
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            depth_observations = preprocess(depth_observations)
            cnn_input.append(depth_observations)

        cnn_input = torch.cat(cnn_input, dim=1)

        with torch.no_grad():
            return self.cnn(cnn_input)
