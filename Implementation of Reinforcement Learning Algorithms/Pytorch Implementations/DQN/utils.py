import numpy as np
import torch
from torch import FloatTensor
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as T


class Utils:

    # This is based on the code from gym.
    ScreenWidth = 600


    def __init__(self):
        self.resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation = Image.CUBIC), T.ToTensor()])

    def numpy_array_to_torch_tensor(self, numpy_array, dtype = np.float32, tensor_type = FloatTensor, is_volatile = False):
        """
        This function converts a numpy array to a pytorch tensor

        :param numpy_array (obj:`numpy array`): the numpy array to be converted
        :param dtype (obj:`numpy float type`): the dtype of the numpy array
        :param tensor_type (obj:`Pytorch Tensor`): the type of the final output tensor
        """

        if numpy_array.dtype != dtype:
            numpy_array = numpy_array.astype(dtype)
        return Variable(torch.from_numpy(numpy_array).type(tensor_type), volatile = is_volatile)


    def get_cart_location(self, rl_environment):
        world_width = rl_environment.x_threshold * 2
        scale = self.ScreenWidth / world_width
        return int(rl_environment.state[0] * scale + self.ScreenWidth / 2.0)  # MIDDLE OF CART

    def get_screen(self, rl_environment):

        # transpose into torch order (CHW)
        screen = rl_environment.render(mode='rgb_array').transpose((2, 0, 1))

        # Strip off the top and bottom of the screen
        screen = screen[:, 160:320]
        view_width = 320
        cart_location = self.get_cart_location(rl_environment)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (self.ScreenWidth - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)

        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0)
