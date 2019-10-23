import numpy as np
import torch
from torch.autograd import Function,Variable

#import ipdb

class GemanMcclureLoss(Function):
    """
    Define Loss based on Geman-Mcclure
    """
    @staticmethod
    def forward(ctx, input, target):
        sigma = 0.1
        x = input-target
        ctx.saved_variable = (x,sigma)
        # ipdb.set_trace()
        return input.new([(x**2 / (x**2 + sigma**2)).sum() / x.nelement()])
        # return (x**2 / (x**2 + sigma**2)).sum() / x.nelement()

    @staticmethod
    def backward(ctx, grad_output=None):
        x,sigma = ctx.saved_variable

        # import ipdb;ipdb.set_trace()
        grad = Variable(2*x*sigma**2 / ((x**2 + sigma**2)**2) / x.nelement())

        return grad*grad_output,None


class AdaptiveGemanMcclureLoss(Function):
    """
    Define Loss based on Geman-Mcclure
    """
    @staticmethod
    def forward(ctx, input, target):
        _mad = lambda x : (x - x.median()).abs().median()
        x = input-target
        sigma = 1.4826 * _mad(x)
        ctx.saved_variable = (x,sigma)
        # ipdb.set_trace()
        return input.new([(x**2 / (x**2 + sigma**2)).sum() / x.nelement()])
        # return (x**2 / (x**2 + sigma**2)).sum() / x.nelement()

    @staticmethod
    def backward(ctx, grad_output=None):
        x,sigma = ctx.saved_variable

        # import ipdb;ipdb.set_trace()
        grad = Variable(2*x*sigma**2 / ((x**2 + sigma**2)**2) / x.nelement())

        return grad*grad_output,None


class EPELoss(Function):
    """
    Loss based on average endpoint error
    """
    @staticmethod
    def forward(ctx, input, target):
        x = input-target
        df = (x**2).sum(dim=1,keepdim=True)
        ctx.saved_variable = (x,df)
        return input.new( [ df.sqrt().sum() / (x.nelement()/2.0) ] )

    @staticmethod
    def backward(ctx, grad_output=None):
        x,df = ctx.saved_variable
        df_stacked = torch.cat((df,df),dim=1)
        grad = Variable( x * df_stacked.rsqrt() / (x.nelement()/2.0) )
        return grad*grad_output, None




def main():
    # Test GemanMcclure criterion
    from torch.autograd import gradcheck

    # gradchek takes a tuple of tensor as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (Variable(torch.randn(20,2).double(), requires_grad=True), Variable(torch.randn(20,2).double(), requires_grad=False),)

    # Test manually
    loss = GemanMcclureLoss.apply(input[0],input[1])
    print('=== Loss ===')
    print(loss)
    loss.backward()
    print('=== Grad ===')
    print(input[0].grad)

    test = gradcheck(GemanMcclureLoss.apply, input, eps=1e-6, atol=1e-4, raise_exception=True)
    print(test)

    # Gradcheck for EPELoss
    input = (Variable(torch.randn(3,2,10,10).double(), requires_grad=True),
             Variable(torch.randn(3,2,10,10).double(), requires_grad=False))
    test = gradcheck(EPELoss.apply, input, eps=1e-6, atol=1e-4, raise_exception=True)
    print(test)



if __name__ == '__main__':
    main()

