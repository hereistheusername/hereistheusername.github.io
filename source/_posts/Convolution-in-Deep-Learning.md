---
title: Convolution in Deep Learning
date: 2023-03-17 23:55:59
tags: conv2d, convolution in deep learning
---
# A misunderstanding about convolution in deep learning

The definition of convolution in deep learning is somehow different from that in math or engineering.
Check this blog [http://www.songho.ca/dsp/convolution/convolution2d_example.html](http://www.songho.ca/dsp/convolution/convolution2d_example.html)

By this definition, before doing element wise product and traversing, we have to **flip** the kernel. However, it doesn't work like this in deep learning.

Let's do an experiment in Pytorch.

First, define a function to help us specify the kernel.


```python
import torch
import torch.nn as nn
```


```python
def new_conv2d_with_kernel(kernel: torch.tensor, **kwargs) -> nn.Conv2d:
    """
    create a 2d convolutional layer with specified kernel for learning convolution operation in deep learning

    :param kernel: one channel kernel
    :param kwargs: named parameters passed to Conv2d
    :return: a convolutional layer which can process 1 channel matrix for 1 batch
    """
    c = nn.Conv2d(1, 1, kernel.shape, **kwargs)
    p = nn.parameter.Parameter(kernel.view(1, 1, *kernel.shape), requires_grad=True)    #Only Tensors of floating point and complex dtype can require gradients
    c.weight = p
    return c
```

see the outcomes.


```python
new_conv2d_with_kernel(torch.tensor([[1,1], [0, 0]], dtype=torch.float)).weight
```




    Parameter containing:
    tensor([[[[1., 1.],
              [0., 0.]]]], requires_grad=True)



So, lets try an example

![cited example](https://miro.medium.com/v2/resize:fit:1052/1*GcI7G-JLAQiEoCON7xFbhg.gif)

cited from: [https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)


```python
input = torch.tensor([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]], dtype=torch.float)
kernel = torch.tensor([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]], dtype=torch.float)
conv2d = new_conv2d_with_kernel(kernel)
conv2d(input.view(1, 1, *input.shape)).type(torch.int)
```




    tensor([[[[3, 2, 3],
              [1, 3, 2],
              [1, 2, 3]]]], dtype=torch.int32)



Let's try another example in a mathematical background.
[http://www.songho.ca/dsp/convolution/convolution2d_example.html](http://www.songho.ca/dsp/convolution/convolution2d_example.html)


```python
input = torch.arange(1, 10).reshape(3, 3).type(torch.float)
kernel = torch.tensor([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]], dtype=torch.float)
conv2d = new_conv2d_with_kernel(kernel, padding=1)
conv2d(input.view(1, 1, *input.shape))
```




    tensor([[[[ 13.0554,  20.0554,  17.0554],
              [ 18.0554,  24.0554,  18.0554],
              [-12.9446, -19.9446, -16.9446]]]], grad_fn=<ConvolutionBackward0>)



The output is different from the example in this blog.

Try what gonna happen if we flip the kernel. (flipping should happen on each axis!)


```python
flipped_kernel = torch.tensor([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]], dtype=torch.float)
conv2d = new_conv2d_with_kernel(flipped_kernel, padding=1)
conv2d(input.view(1, 1, *input.shape))
```




    tensor([[[[-13.1509, -20.1509, -17.1509],
              [-18.1509, -24.1509, -18.1509],
              [ 12.8491,  19.8491,  16.8491]]]], grad_fn=<ConvolutionBackward0>)



This time the output matches the example. And you can check this [post](https://cs.stackexchange.com/questions/11591/2d-convolution-flipping-the-kernel) to see the consequence of misusing.

# Conclusion

Concepts in different subjects may share the same name but with different definitions. Be careful with that.

