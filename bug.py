import lightning as L
import torch.nn as nn


class a(L.LightningModule):
    def __init__(self, testing=True):
        super(a, self).__init__()
        if testing:
            self.model = nn.Linear(2, 4)
        else:
            self.model = nn.Linear(4, 8)


model = a(False)

trainer = L.Trainer()
trainer.strategy.connect(model)
trainer.save_checkpoint("bugtesting.ckpt")

model = a.load_from_checkpoint(
    "bugtesting.ckpt", testing=False
)  # comment this one and uncomment bottom two for more weird behavior

# model2 = a(False)
# model2.load_from_checkpoint("bugtesting.ckpt") #This should error out, but it doesn't.....
