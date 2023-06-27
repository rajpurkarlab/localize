import lightning.pytorch as pl
import open_clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_biomed_clip(device):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.to(device)
    return model, tokenizer, preprocess_train, preprocess_val

# define the LightningModule
class Model(pl.LightningModule):
    def __init__(self, lseg):
        super().__init__()
        self.lseg = lseg
        self.biomed_clip_model, self.biomed_clip_tokenizer, self.preprocess_train, self.preprocess_val = load_biomed_clip(device)
        # image_features, text_features, logit_scale = biomed_clip_model(images, texts)

    def train_dataloader(self):
        loader_lseg = self.lseg.train_dataloader()
        loader_biomed_clip = None # load the biomed clip data

        return {"lseg": loader_lseg, "biomed_clip": loader_biomed_clip}

    def training_step(self, batch, batch_idx):
        # access a dictionary with a batch from each DataLoader
        batch_lseg = batch["lseg"]
        batch_biomed_clip = batch["biomed_clip"]

        seg_loss = self.lseg.training_step(batch_lseg, batch_idx)
        adapt_loss = adapt_loss(batch_biomed_clip)

    def adapt_loss(self, batch_biomed_clip):
        # get the image and text features from biomed clip
        # get the image and text features from lseg
        # compute the loss between the two