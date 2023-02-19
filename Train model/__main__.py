from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score
import argparse
import json


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--path-to-model", required=True, type=str)
    parser.add_argument("--path-to-math-dataset", required=True, type=str)
    parser.add_argument("--path-to-biology-dataset", required=True, type=str)
    return parser.parse_args()


class BertDataset(Dataset):
    def __init__(self, corpus, labels, tokenizer):
        super().__init__()
        self._corpus = []
        self._labels = labels
        self._tokenizer = tokenizer

        for word in corpus:
            self._corpus.append(tokenizer(word).input_ids)

    def __len__(self):
        return len(self._corpus)

    def __getitem__(self, idx):
        return self._corpus[idx], self._labels[idx]


class Collator:
    def __init__(self, tokenizer):
        self._pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        inputs = [torch.tensor(x[0]) for x in batch]
        input_ids = pad_sequence(
            inputs,
            padding_value=self._pad_token_id,
            batch_first=True
        )
        labels = torch.tensor([x[1] for x in batch])

        return input_ids, labels


def init_layer(layer, initializer_range=0.02, zero_out_bias=True):
    if isinstance(layer, nn.Embedding) or isinstance(layer, nn.Linear):
        nn.init.trunc_normal_(layer.weight.data, std=initializer_range, a=-2 * initializer_range, b=2 * initializer_range)
        if isinstance(layer, nn.Linear):
            layer.bias.data.zero_()


class BertModel(nn.Module):
    CLS_POSITION = 0

    def __init__(self, bert_backbone, hidden_size, num_classes=1, hidden_dropout_prob=0.):
        super().__init__()
        self._bert_backbone = bert_backbone

        # freeze
        for param in self._bert_backbone.parameters():
            param.requires_grad = False

        self._hidden_size = hidden_size
        self.linear_1 = nn.Linear(1024, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, num_classes)
        self.model = nn.Sequential(
            self.linear_1,
            nn.ReLU(),
            nn.Dropout(p=hidden_dropout_prob),
            self.linear_2
        )

        # for layer in [self.linear_1, self.linear_2]:
        #     init_layer(layer)

    def forward(self, x, attn_mask):
        bert_cls_output = self._bert_backbone(x, attn_mask).last_hidden_state[:, self.CLS_POSITION]
        return self.model(bert_cls_output)


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            pad_token_id,
            device="cpu",
    ):
        model.to(device)
        self.model = model
        self.optimizer = optimizer
        self.pad_token_id = pad_token_id
        self.device = device
        self.loss = nn.BCEWithLogitsLoss()

    def train(self, dataloader, n_epochs):
        for epoch in range(n_epochs):
            train_loss = self._train_step(dataloader)
            val_loss, val_acc, val_auc = self._eval_step(dataloader)
            print(
                f'epoch: {epoch}, train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}, val_auc: {val_auc:.3f}')

    def _train_step(self, dataloader):
        self.model.train()
        batch_loss = 0
        for _, (input_ids, labels) in enumerate(dataloader):
            self.optimizer.zero_grad()
            input_ids = input_ids.to(self.device)
            attention_mask = (input_ids != self.pad_token_id).float().to(self.device)
            outputs = self.model(input_ids, attention_mask).to("cpu").flatten()
            loss = self.loss(outputs, labels)
            batch_loss += loss.float()
            loss.backward()
            self.optimizer.step()

        return batch_loss / len(dataloader)

    def _eval_step(self, dataloader):
        self.model.eval()
        labels_lst = []
        pred_lst = []
        with torch.no_grad():
            batch_loss = 0
            acc = 0
            num_objects = 0
            for i, (input_ids, labels) in enumerate(dataloader):
                input_ids = input_ids.to(self.device)
                attention_mask = (input_ids != self.pad_token_id).float().to(self.device)

                outputs = self.model(input_ids, attention_mask).to("cpu").flatten()
                loss = self.loss(outputs, labels)

                outputs = torch.nn.Sigmoid()(outputs)

                binary_outputs = (outputs > 0.5).float()

                batch_loss += loss.float()
                acc += (binary_outputs == labels).sum()

                labels_lst.extend(labels.tolist())
                pred_lst.extend(binary_outputs.tolist())

                num_objects += input_ids.shape[0]

            return batch_loss / len(dataloader), acc / num_objects, roc_auc_score(labels_lst, pred_lst)


def get_dataloader(corpus, targets, tokenizer):
    ds = BertDataset(
        corpus=corpus,
        labels=targets,
        tokenizer=tokenizer,
    )

    collator = Collator(tokenizer)

    dl = DataLoader(
        ds,
        collate_fn=collator,
        batch_size=16,
        # shuffle=False
        shuffle=True
    )

    return dl


def main():
    args = argparser()

    model = AutoModel.from_pretrained(args.path_to_model)
    tokenizer = AutoTokenizer.from_pretrained(args.path_to_model)

    with open(args.path_to_math_dataset, mode='r') as f:
        math_json = json.load(f)
    with open(args.path_to_biology_dataset, mode='r') as f:
        biology_json = json.load(f)

    corpus = math_json + biology_json
    targets = [0.] * len(math_json) + [1.] * len(biology_json)
    targets = list(map(float, targets))

    dl = get_dataloader(corpus, targets, tokenizer)

    bert_model = BertModel(model, 16)

    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=1e-3)

    trainer = Trainer(bert_model, optimizer, tokenizer.pad_token_id)

    trainer.train(dl, 100)

    return


if __name__ == "__main__":
    main()
