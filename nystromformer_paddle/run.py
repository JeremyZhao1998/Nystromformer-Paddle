import paddle
from paddle.io import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import pickle

dataset = 'imdb'
max_len = 512
batch_size = 2
device = 'cuda:0'
lr = 1e-4
epochs = 100
mixed_precision = True


def prepare_loader(split):
    data_path = '../data/tokenized_' + dataset + '_' + split + '_' + str(max_len) + '.pkl'
    try:
        with open(data_path, 'rb') as f:
            input_ids, token_type_ids, attention_mask, labels = pickle.load(f)
    except FileNotFoundError:
        raw_data = load_dataset(dataset)
        tokenizer = AutoTokenizer.from_pretrained('../pretrained_files')
        tokenized_data = tokenizer(
            raw_data[split]['text'],
            truncation=True, padding='max_length', max_length=max_len - 2,
            return_tensors='np'
        )
        f = open(data_path, 'wb')
        input_ids, token_type_ids, attention_mask = \
            tokenized_data['input_ids'], tokenized_data['token_type_ids'], tokenized_data['attention_mask']
        labels = raw_data[split]['label']
        pickle.dump((input_ids, token_type_ids, attention_mask, labels), f)

    class TextDataset(Dataset):
        def __len__(self):
            return len(input_ids)

        def __getitem__(self, idx):
            return input_ids[idx], token_type_ids[idx], attention_mask[idx], labels[idx]

    loader = DataLoader(dataset=TextDataset(), batch_size=batch_size, shuffle=split == 'train')
    return loader


def main():

    train_loader = prepare_loader('train')

    for batch_data in train_loader:
        print(batch_data)
        break

    """for epoch in range(epochs):
        label_list, predict_list = [], []
        for data in train_loader:
            input_ids, attention_mask, labels = data
            input_ids, attention_mask = input_ids[:, 1: -1].to(device), attention_mask[:, 1: -1].to(device)
            labels = labels.unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            label_list.extend(labels.cpu().data.tolist())
            predict_list.extend(outputs.logits.argmax(-1).cpu().data.tolist())
            loss = outputs.loss
            if amp_scaler is not None:
                scaled = amp_scaler.scale(loss)
                scaled.backward()
                amp_scaler.minimize(optimizer, scaled)
            else:
                loss.backward()
                optimizer.minimize(loss)
            model.clear_gradients()
            lr_scheduler.step()
            print(loss)
        acc = accuracy_score(y_true=label_list, y_pred=predict_list)
        f1 = f1_score(y_true=label_list, y_pred=predict_list, average='binary')
        print(epoch, acc, f1)"""


if __name__ == '__main__':
    main()
