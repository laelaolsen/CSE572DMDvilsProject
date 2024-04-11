import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv

# Function to read CSV file and extract columns
def read_csv(file_path):
    columns = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for column in reader.fieldnames:
            columns[column] = []
        for row in reader:
            for column in reader.fieldnames:
                columns[column].append(row[column])
    return columns

# Define the BERT model
class BERTClassifier(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0]

# Custom Dataset class for handling data
class CustomDataset(Dataset):
    def __init__(self, descriptions, labels, tokenizer, max_len):
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        description = str(self.descriptions[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'description': description,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Function to train the model
def train(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Function to evaluate the model
def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            val_preds.extend(preds.tolist())
            val_labels.extend(labels.tolist())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)

    return val_loss, val_acc

if __name__ == "__main__":

    gpt_separate = read_csv("C:/Users/laela/Downloads/Separate_Pairs_FINAL_Result_wGPT.csv")["GPT"]
    cagpt_separate = read_csv("C:/Users/laela/Downloads/Separate_Pairs_FINAL_Result_wGPT.csv")["CAGPT"]
    gpt_mixed = read_csv("C:/Users/laela/Downloads/Mixed_Pairs_FINAL_Result_wGPT.csv")["GPT"]
    cagpt_mixed= read_csv("C:/Users/laela/Downloads/Mixed_Pairs_FINAL_Result_wGPT.csv")["CAGPT"]

    # Combine the GPT results
    gpt = gpt_separate + gpt_mixed
    cagpt = cagpt_separate + cagpt_mixed

    labels = []
    for i in range(len(gpt)):
        labels.append(0)
    for i in range(len(cagpt)):
        labels.append(1)
    
    # Split the data into training and validation sets
    train_descriptions, val_descriptions, train_labels, val_labels = train_test_split(
        gpt + cagpt, labels, test_size=0.2, random_state=42
    )

    # Tokenize the descriptions
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 256  # Max length of input tokens
    train_dataset = CustomDataset(train_descriptions, train_labels, tokenizer, max_len)
    val_dataset = CustomDataset(val_descriptions, val_labels, tokenizer, max_len)

    # Define data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERTClassifier(num_classes=2).to(device)

    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Train the model
    train(model, train_loader, val_loader, optimizer, loss_fn, device)
