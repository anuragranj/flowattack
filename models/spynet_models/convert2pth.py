import glob
import torch.utils.serialization

files = glob.glob('*.t7')

for f in files:
    weight = torch.utils.serialization.load_lua(f)
    torch.save(weight, f[:-3]+'.pth.tar')
    print('saving', f)
