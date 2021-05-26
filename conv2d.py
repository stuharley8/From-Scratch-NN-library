import torch


def conv_filter_forward(image,filter,padding=0):
    """
    Convolves an m-channel image with a bank of n filters each with m channels
    to produce an n-channeled image.  To produce the nth channel, the nth filter
    is convolved with the image.

    :param image: m x h x w torch tensor. m is the number of channels
    :param filter: n x m x c0 x c1 torch tensor. n is the number of filters
                                            and m is the number of channels in each filter
    :param padding: pixels to add on each edge of each channel of the input image.  Pads the input
                    to be m x (h+2pad) x (w+2pad) where pad is the padding argument.
    :return: n x h+2pad-s0+1 x w+2pad-s1+1 tensor resulting from convolving image with filter.
    """
    assert image.dtype is torch.float32
    assert filter.dtype is torch.float32
    assert len(image.shape) == 3
    assert len(filter.shape) == 4
    num_chan_image = image.shape[0]
    h = image.shape[1] # dimensions of image
    w = image.shape[2]
    num_filters = filter.shape[0]
    assert filter.shape[1] == num_chan_image
    s0 = filter.shape[2] # dimensions of filter
    s1 = filter.shape[3]
    image = image.reshape((1,num_chan_image,h,w))
    conv = torch.nn.Conv2d(in_channels=num_chan_image,out_channels=num_filters,
                           kernel_size=(s0,s1),bias=False,padding=padding)
    conv.requires_grad_(False)
    assert conv.weight.data.shape == (num_filters,num_chan_image,s0,s1)
    assert conv.weight.data.shape == filter.shape
    conv.weight.data = filter
    map = conv(image)
    assert map.shape == (1,num_filters,h+2*padding-s0+1,w+2*padding-s1+1)
    map = map.reshape(num_filters,h+2*padding-s0+1,w+2*padding-s1+1)
    return map


def conv_filter_backward(image, filter, padding=0):
    """
    Convolves an n-channel image with a bank of n filters each with m channels
    to produce an m-channeled image.  To produce the mth channel, the mth channel
    of each filter is convolved with the corresponding channel of the image
    and the result is summed into a single channel.

    This is "backwards" because in forward convolution, we would convolve
    an m-channel image with a bank of n filters each with m channels to produce
    an n-channel image.

    :param image: n x h x w torch tensor. n is the number of channels
    :param filter: n x m x c0 x c1 torch tensor. n is the number of filters
                                            and m is the number of channels in each filter
    :param padding: pixels to add on each edge of each channel of the input image.  Pads the input
                    to be n x (h+2pad) x (w+2pad) where pad is the padding argument.
    :return: m x h+2pad-s0+1 x w+2pad-s1+1 tensor resulting from convolving image with filter.
    """
    assert image.dtype is torch.float32
    assert filter.dtype is torch.float32
    assert len(image.shape) == 3
    assert len(filter.shape) == 4
    num_chan_image = image.shape[0]
    h = image.shape[1] # dimensions of image
    w = image.shape[2]
    assert filter.shape[0] == num_chan_image
    num_chan_filter = filter.shape[1]
    s0 = filter.shape[2] # dimensions of filter
    s1 = filter.shape[3]
    image = image.reshape((1,num_chan_image,h,w))
    filter = filter.permute((1,0,2,3))
    conv = torch.nn.Conv2d(in_channels=num_chan_image,out_channels=num_chan_filter,
                           kernel_size=(s0,s1),bias=False,padding=padding)
    conv.requires_grad_(False)
    assert conv.weight.data.shape == (num_chan_filter,num_chan_image,s0,s1)
    assert conv.weight.data.shape == filter.shape
    conv.weight.data = filter
    map = conv(image)
    assert map.shape == (1,num_chan_filter,h+2*padding-s0+1,w+2*padding-s1+1)
    return map


def conv_expand_layers(image,filter,padding=0):
    """
    Convolves an n-channel image with a m-channel filter,
    producing mxn channels in the output, one for every input channel convolved
    with every filter channel

    :param image: n x h x w torch tensor. n is the number of channels
    :param filter: m x s0 x s1 torch tensor. m is the number of channels
    :param padding: pixels to add on each edge of each channel of the input image.  Pads the input
                    to be n x (h+2pad) x (w+2pad) where pad is the padding argument.
    :return: n x m x h+2pad-s0+1 x w+2pad-s1+1 tensor resulting from convolving image with filter.
    """
    assert image.dtype is torch.float32
    assert filter.dtype is torch.float32
    assert len(image.shape) == 3
    assert len(filter.shape) == 3
    num_chan_image = image.shape[0]
    h = image.shape[1] # dimensions of image
    w = image.shape[2]
    num_chan_filter = filter.shape[0]
    s0 = filter.shape[1] # dimensions of filter
    s1 = filter.shape[2]
    filter = filter.reshape((num_chan_filter,1,s0,s1))
    image = image.reshape((num_chan_image,1,h,w))
    conv = torch.nn.Conv2d(in_channels=1,out_channels=num_chan_filter,
                           kernel_size=(s0,s1),bias=False,padding=padding)
    conv.requires_grad_(False)
    assert conv.weight.data.shape == (num_chan_filter,1,s0,s1)
    assert conv.weight.data.shape == filter.shape
    conv.weight.data = filter
    map = conv(image)
    assert map.shape == (num_chan_image,num_chan_filter,h+2*padding-s0+1,w+2*padding-s1+1)
    map = map.permute((1,0,2,3))
    assert map.shape == (num_chan_filter,num_chan_image,h+2*padding-s0+1,w+2*padding-s1+1)
    return map
