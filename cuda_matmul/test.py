import torch
import cppexm


if __name__=="__main__":
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    b = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
    c = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

    a = torch.ones((15, 15))
    b = torch.ones((15, 15))
    c = torch.ones((15, 15))
    a = a.cuda()
    b = b.cuda()
    c = c.cuda()
    cppexm.matmul_cuda(a, b, c)
    print(c)