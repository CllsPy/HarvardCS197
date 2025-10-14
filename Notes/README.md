# CS197 - Machine Learning & Deep Learning Study Guide

**Criado em:** 30 de julho de 2025  
**Tags:** AI, Courses, Machine Learning, AI Research



## Metodologia de Estudo

> *"Go through the material. Did it click? Write down what helped, otherwise look for a better explanation."*

**Princípio Principal:** Building is the most effective way of learning when it comes to AI/ML engineering.



## Aula 1: Tensores com PyTorch

### Conceitos Fundamentais

**Tensor** é um array multidimensional que pode ser processado em CPU ou GPU. É a estrutura fundamental dos modelos de deep learning.

### Exercícios Práticos

**1. Criar um tensor a partir de lista aninhada**
```python
data = [[5, 3], [0, 9]]
x_data = torch.tensor(data)
```

**2. Criar tensor com números aleatórios (distribuição uniforme [0,1))**
```python
t = torch.rand((5, 4))
```

**3. Verificar device e dtype**
```python
print(t.device)  # cpu
print(t.dtype)   # float32
```

**4. Concatenar tensores (shape (8, 4))**
```python
u = torch.randn((4, 4))
v = torch.randn((4, 4))
result = torch.concat((u, v), dim=0)  # shape: [8, 4]
```

**5. Stack tensores (criar nova dimensão)**
```python
w = torch.stack((u, v), dim=2)  # shape: [4, 4, 2]
```

**6. Indexação**
```python
e = w[3, 3, 0]
```

### Stack vs Cat

| Operação | O que faz | Resultado |
|----------|----------|-----------|
| `stack()` | Cria uma **nova dimensão** | Aumenta dimensionalidade |
| `cat()`   | Usa **dimensão existente** | Mantém dimensões, concatena |

---

## Aula 2: Ambiente de Desenvolvimento

### Atalhos Essenciais do VSCode

- `Ctrl+Shift+P` - Abrir Command Palette
- `Ctrl+B` - Retrair/expandir sidebar
- `Ctrl+K Z` - Modo foco (tela cheia)
- `Ctrl+Shift+'` - Abrir terminal
- `Ctrl+Shift+M` - Ir ao erro
- `F8` - Navegar entre erros
- `Ctrl+K Ctrl+F` - Formatar código
- `Shift+Alt+F` - Autoformat
- `Ctrl+Space` - Sugestões de código

### Configuração Inicial de Projeto

```bash
mkdir hello           # Criar pasta
cd hello              # Entrar na pasta
code .               # Abrir no VSCode
```

### Ambientes Virtuais

Um venv isola dependências de um projeto específico, evitando conflitos.

**Com venv (Python padrão):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Com Conda:**
```bash
conda create -p venv python==3.12    # No diretório do projeto
conda create -n venv python==3.12    # Na pasta do conda
conda activate venv/
```

### Conceitos Importantes para Aprofundar
- Criar e gerenciar venvs
- Debugging no VSCode
- `pip freeze` para gerenciar pacotes
- Ferramentas: `uv`, `conda`

---

## Aula 4: Fine-tuning de Modelos de Linguagem

### Objetivo de Aprendizagem

- Dataset loading
- Tokenization
- Fine-tuning para causal language modeling

### Instalação de Dependências

```bash
pip install transformers datasets
```

### Passo-a-Passo: Fine-tuning com SQuAD

**1. Carregar dataset**
```python
from datasets import load_dataset
dataset = load_dataset("nunorc/squad_v1_pt")
```

**2. Processar dataset**
```python
def add_end_of_text(example):
    example['question'] = example['question'] + '<|endoftext|>'
    return example

dataset = dataset.remove_columns(['id', 'title', 'context', 'answers'])
dataset = dataset.map(add_end_of_text)
```

O token `<|endoftext|>` marca o fim de uma sequência quando múltiplas são concatenadas.

**3. Carregar tokenizer**
```python
from transformers import AutoTokenizer
model_checkpoint = "tubyneto/bertimbau"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
```

**4. Tokenizar textos**
```python
sequence = "Está tokenização está sendo aplicada em pt-br. Por CLL.<|endoftext|>"
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
```

A tokenização converte texto em IDs numéricos que o modelo compreende.

**5. Processar dataset em batches**
```python
def tokenize_function(examples):
    return tokenizer(examples["question"], truncation=True)

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["question"]
)
```

**6. Agrupar textos por blocos**
```python
block_size = 128

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    
    result = {
        k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=4)
```

**7. Preparar dados de treino e validação**
```python
small_train_dataset = lm_datasets["train"].shuffle(seed=42).select(range(100))
small_eval_dataset = lm_datasets["validation"].shuffle(seed=42).select(range(100))
```

**8. Configurar e treinar modelo**
```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

training_args = TrainingArguments(
    f"{model_checkpoint}-squad",
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)

trainer.train()
```

**9. Avaliar modelo**
```python
import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

**10. Salvar e fazer upload**
```python
tokenizer.save_pretrained(f"{model_checkpoint}-squad")
model.push_to_hub(f"{model_checkpoint}-squad")
```

**11. Fazer inferência**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(f"rajpurkar/{model_checkpoint}-squad")
tokenizer = AutoTokenizer.from_pretrained(f"rajpurkar/{model_checkpoint}-squad")

start_text = "A speedrun é uma jogada de videogame com o objetivo de completá-la o mais rápido possível."
prompt = "O que é"

inputs = tokenizer(start_text + prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

outputs = model.generate(
    inputs,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    num_return_sequences=3
)

print(tokenizer.decode(outputs[0]))
```

### Tópicos para Aprofundamento

- Fine-tuning com DeepSeek no SQuAD-PT-V1
- Biblioteca `unsloth` para otimização

---

## Aula 5: Vision Transformers (ViT)

### Objetivo de Aprendizagem

- Carregar e processar datasets de imagens
- Compreender arquitetura de Vision Transformers
- Workflow de treinamento com PyTorch Lightning

### PyTorch Lightning

PyTorch Lightning simplifica o treinamento abstraindo engenharia complexa. Facilita organização do código e otimização de hiperparâmetros.

### Estrutura Básica (AutoEncoder com Lightning)

```python
import pytorch_lightning as pl
import torch.nn.functional as F

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

### Processamento de Imagens

**1. Transformações de treino**
```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.49, 0.48, 0.45], [0.25, 0.24, 0.26]),
])
```

No treino usamos augmentação. No teste, apenas normalização para acelerar.

**2. Carregar dataset CIFAR-10**
```python
from torchvision.datasets import CIFAR10

train_dataset = CIFAR10(root="data/", train=True, transform=train_transform, download=True)
test_set = CIFAR10(root="data/", train=False, transform=test_transform, download=True)
```

**3. Preparar dataloaders**
```python
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=16, shuffle=True, num_workers=0, pin_memory=True
)
```

### Tokenização de Imagens em Patches

Imagens são convertidas em sequências de patches (similar a tokenização de texto):

```python
def img_to_patch(x, patch_size, flatten_channels=True):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.flatten(1, 2)
    if flatten_channels:
        x = x.flatten(2, 4)
    return x
```

### Arquitetura: Attention Block

```python
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
```

### Modelo Vision Transformer Completo

```python
class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, 
                 num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size
        
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
              for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)
        
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, :T+1]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        cls = x[0]
        return self.mlp_head(cls)
```

### Treinador com PyTorch Lightning

```python
class ViT(pl.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
```

### Treinar Modelo

```python
def train_model(**kwargs):
    trainer = pl.Trainer(
        default_root_dir="saved_models/VisionTransformers/",
        fast_dev_run=5,
    )
    
    pl.seed_everything(42)
    model = ViT(**kwargs)
    trainer.fit(model, train_loader, val_loader)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    return model, test_result

model, results = train_model(
    model_kwargs={
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "patch_size": 4,
        "num_channels": 3,
        "num_patches": 64,
        "num_classes": 10,
        "dropout": 0.2,
    },
    lr=3e-4,
)
```


## Aulas 6-7: PyTorch Fundamentals

### Conceitos Essenciais

1. **Tensor** - Matriz multidimensional processável em GPU
2. **Autograd** - Sistema automático de cálculo de gradientes
3. **Computational Graph** - Mapa de operações para retropropagação
4. **GPU Acceleration (CUDA)** - Processamento paralelo em placa gráfica
5. **Modules (nn.Module)** - Blocos de construção de redes neurais
6. **Layers** - Transformações prontas (convolução, linear, etc)
7. **Loss Functions** - Medidores de erro
8. **Optimizers** - Algoritmos para ajustar pesos
9. **Dataset e DataLoader** - Carregadores de dados em batches
10. **Training Loop** - Ciclo: dados → predição → erro → ajuste
11. **Inference Mode** - Modo sem cálculo de gradientes
12. **Save/Load Models** - Persistência de modelos

### Verificar GPU

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Mover tensor para GPU
tensor = torch.randn(1000, 1000).to(device)
```

**CUDA** permite que a GPU processe múltiplos tensores em paralelo, acelerando exponencialmente as operações.

### Broadcasting

Quando dois tensores de shapes diferentes interagem, PyTorch "estica" automaticamente as dimensões menores. Exemplo: shape `(4, 3)` e `(3,)` podem operar juntas.

**Nota:** O broadcasting não aloca memória extra; é apenas uma operação lógica.

### Exercícios Fundamentais

**Multiplicação elemento-wise:**
```python
a = torch.ones((4, 3))
result = a * a  # Multiplica cada elemento por si mesmo
```

**Multiplicação matricial:**
```python
result = a @ a.T  # Produto escalar de a com sua transposta
```

**Conversão para NumPy (compartilha memória):**
```python
t = torch.ones(5)
n = t.numpy()
n[0] = 2
print(t)  # Mudanças refletem em ambos
```


## Treinamento de Redes Neurais: Forward e Backward Pass

### Forward Pass

A rede neural faz previsões tentando aproximar os dados reais o máximo possível.

### Backward Pass

Compara a previsão com o valor real, calcula o erro (loss) e ajusta os pesos para reduzir esse erro através da retropropagação (backpropagation).

### Ciclo Típico de Treinamento

1. Iterar sobre dados
2. Forward pass (predição)
3. Calcular erro (loss)
4. Backward pass (calcular gradientes)
5. Atualizar pesos com optimizer
6. Repetir



## Recursos Adicionais

- [PyTorch Official Documentation](https://pytorch.org/docs)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
- [VSCode Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [PyTorch Cheat Sheet](https://www.simonwenkel.com/publications/cheatsheets/pdf/cheatsheet_pytorch.pdf)
- [Developing Intuition for Math](https://betterexplained.com/articles/developing-your-intuition-for-math/)
- [Machine Learning Course](https://betterexplained.com/articles/adept-machine-learning-course/)



## Tópicos para Aprofundamento Futuro

- [ ] Fine-tuning com DeepSeek e Unsloth
- [ ] Test-time augmentation para visão computacional
- [ ] Debugging avançado no VSCode
- [ ] Gerenciamento de ambientes com `uv` e `conda`
- [ ] Deploy de modelos em produção
- [ ] Técnicas de otimização (quantization, pruning)
- [ ] Multi-GPU training com PyTorch Lightning
