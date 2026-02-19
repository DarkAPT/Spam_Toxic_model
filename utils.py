from typing import List
from matplotlib import pyplot as plt
from pandas import DataFrame

import re
import numpy as np
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn


class DataPreprocessor:
    def __init__(self) -> None:
        try:
            self.russian_stopwords = set(stopwords.words("russian"))
        except LookupError:
            print("NLTK stopwords not found. Please run: nltk.download('stopwords')")
            self.russian_stopwords = set()
        
        self.morph = MorphAnalyzer()


    def all_preprocess(self, text:str, lemmatize:bool = False, stopwords:bool = True, light_preprocess:bool = False) -> str:
        """
        Основной метод предобработки текста
        
        Args:
            text: исходный текст
            lemmatize: применять лемматизацию
            remove_stopwords: удалять стоп-слова
            light_preprocess: легкая очистка (URL, хештеги) или полная
        
        Returns:
            обработанный текст
        """
        if not isinstance(text,str) or not text.strip():
            return ""

        if light_preprocess:
            text = self._light_preprocess(text)
        else:
            text = self._full_preprocess(text)
        
        if lemmatize:
            text = self._lemmatize_words(text)
        
        if stopwords:
            text = self._remove_stopwords(text)

        return text

    def _light_preprocess(self, text:str) -> str:
        """Легкая очистка (сохраняет пунктуацию)"""
        text = re.sub(r'http\S+',' URL ',text.lower())
        text = re.sub(r'@\w+', ' MENTION ',text)
        text = re.sub(r'#\w+', ' HASHTAG ',text)
        text = re.sub(r'\s+', ' ',text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[\+]?[7-8]?[\(]?[9]\d{2}[\)]?\d{3}[\-]?\d{2}[\-]?\d{2}', ' PHONE_TOKEN ', text)
        text = re.sub(r'\b\d+\b', ' NUMBER_TOKEN ', text)

        return text.strip()
    
    def _full_preprocess(self, text: str) -> str:
        """Полная очистка (удаляет всю пунктуацию)"""
        text = re.sub(r"([,.?!:;])", r" \1 ", text)
        text = re.sub(r"[^а-яё\s]", "", text.lower())
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _lemmatize_words(self, text:str) -> str:
        """Лемматизация текста"""
        words = [
            self.morph.parse(word)[0].normal_form
            for word in text.split()
        ]
        return ' '.join(words).strip()
    
    def _remove_stopwords(self, text:str):
        """Удаление стоп-слов"""
        words = [
            word for word in text.split()
            if word not in self.russian_stopwords
        ]
        return ' '.join(words).strip()
    

class FeatureExtractor:
    def __init__(self) -> None:
        pass

    def extract_all_features(self, text:str) -> dict:
        """
        Основной метод, который собирает все фичи
        
        args:
        text: текст сообщения

        Returns:
        features: словарь со всеми признаками в тексте
        """
        features = {}
        features.update(self._structural_features(text))
        features.update(self._stylistic_features(text))

        return features

    def _structural_features(self, text:str) -> dict:
        """Структурные features сообщения"""
        return{
            "text_length": len(text),
            "word_count": len(text.split()),
            "has_url": int("http://" in text or "www." in text or "https://" in text),
            "digit_count": sum(char.isdigit() for char in text),
            "special_char_count":sum(not char.isalnum() and not char.isspace() for char in text)
        }
    
    def _stylistic_features(self, text:str) -> dict:
        """Стилистические features сообщения"""
        return{
            "has_phone_number": int(bool(re.search(r'(\+7|8)\s?\(?\s?9\d{2}\s?\)?\s?\d{3}\s?-?\s?\d{2}\s?-?\s?\d{2}', text))),
            "has_email": int(bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)))
        }

def search_best_estimator(pipeline:Pipeline, param_grid:dict, x,y):
    "GridSearchCV пайплайн"
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    temp = grid_search.fit(x,y)
    print("Лучшие параметры:", grid_search.best_params_)
    print("Лучшая оценка f1:", grid_search.best_score_)
    return temp.best_estimator_

def plot_confusion_matrix(model, y_test, y_pred):
    cm = confusion_matrix(y_test,y_pred)
    cmp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    cmp.plot(cmap='Blues')
    plt.show()


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def df_with_metrics(model_name,y_pred, y_true):
    accur = accuracy_score(y_pred=y_pred, y_true=y_true)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    prec = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_pred=y_pred, y_true=y_true)
    roc_auc = roc_auc_score(y_true=y_true,y_score=y_pred)
    return DataFrame({"Model":[model_name], "accuracy_score":[accur],"f1_score":[f1], "precision_score":[prec],"recall_score":[recall], "roc_auc_score":[roc_auc]})


def collate_with_padding(input_batch: List[List[int]], pad_id, device='cpu', lengths=True) -> torch.Tensor:
    """
    Collate функция для DataLoader: padding последовательностей до макс. длины в батче
    Возвращает тензоры 'ids', 'label' и опционально 'lengths'
    """
    seq_len = [len(x['message']) for x in input_batch]
    max_seq_len = min(max(seq_len), 256)
    
    batch = []
    for sequence in input_batch:
        padded_seq = sequence['message'][:max_seq_len]
        padded_seq = padded_seq + [pad_id] * (max_seq_len - len(padded_seq))
        batch.append(padded_seq)

    sequences = torch.LongTensor(batch).to(device)
    labels = torch.FloatTensor([x['label'] for x in input_batch]).to(device)

    sample = {
        'ids': sequences,
        "label": labels.unsqueeze(1),
        
    }

    if lengths:
        lengths = [item['length'] for item in input_batch]
        sample["lengths"] = torch.LongTensor(lengths).to(device)

    return sample

def evaluate_model(model,test_loader,criterion,device='cpu',rnn=False):
    """
    Оценка модели на тестовых данных
    Вычисляет метрики классификации: accuracy, precision, recall, f1, roc_auc

    Параметры:
        model: модель для оценки
        test_loader: DataLoader с тестовыми данными
        criterion: функция потерь
        device: устройство для вычислений ('cpu' или 'cuda')
        rnn: флаг, указывающий что модель рекуррентная (использует lengths)

    Возвращает:
        results: словарь с метриками (accuracy, precision, recall, f1, roc_auc, loss, confusion_matrix)
        all_prob: массив вероятностей для всех примеров
    """
    model.eval()
    total_loss = 0
    all_pred = []
    all_labels = []
    all_prob = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch['label'].to(device)
            if not rnn:
                outputs = model(
                    text_x=batch['text'].to(device),
                    features_x=batch['features'].to(device), 
                )
            else:
                outputs = model(
                    batch['ids'].to(device),
                    batch['lengths'].to(device)
                    )

            loss = criterion(outputs,labels)
            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs).cpu().numpy()
            all_prob.append(probabilities)

            predictions = (probabilities>0.5).astype(int)
            all_pred.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    roc_auc = roc_auc_score(y_true=all_labels,y_score=all_pred)
    accuracy = accuracy_score(all_labels, all_pred)
    precision = precision_score(all_labels, all_pred, average='binary')
    recall = recall_score(all_labels, all_pred, average='binary')
    f1 = f1_score(all_labels, all_pred, average='binary')
    avg_loss = total_loss / len(test_loader)

    cm = confusion_matrix(all_labels, all_pred)

    all_prob = np.concatenate(all_prob)

    results = {
        'accuracy': round(accuracy, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1_score': round(f1, 3),
        "roc_auc_score":round(roc_auc,3),
        'loss': round(avg_loss, 3),
        'confusion_matrix': cm,
    }

    return results, all_prob

class DenseNN(nn.Module):
    """
    Гибридная модель для задач классификации с текстовыми и числовыми признаками

    Параметры:
        vocab_size: размер словаря
        embedding_dim: размерность эмбеддингов слов
        features_dim: количество дополнительных признаков
        hidden_sizes: список размеров скрытых слоев (по умолчанию [256,128,64])
        dropout: вероятность dropout для регуляризации (по умолчанию 0.3)

    Вход:
        text_x: тензор индексов слов [batch_size, seq_len]
        features_x: тензор признаков [batch_size, features_dim]

    Выход:
        логиты для бинарной классификации [batch_size, 1]
    """
    def __init__(self,vocab_size:int,embedding_dim:int,features_dim: int, hidden_sizes:list = [256,128,64], dropout:float = 0.3) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.feature_norm = nn.LayerNorm(features_dim)

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        layers = []
        prev_size = embedding_dim + features_dim

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.extend([
            nn.Linear(prev_size,1),
        ])

        self.network = nn.Sequential(*layers)
    
    def forward(self, text_x, features_x):
        embedded = self.embedding(text_x)
        attention_weights = torch.softmax(self.attention(embedded).squeeze(-1), dim=1)
        text_output = torch.sum(embedded * attention_weights.unsqueeze(-1), dim=1)

        normalized_features = self.feature_norm(features_x)
        
        combined = torch.cat([text_output, normalized_features], dim=1)
        
        return self.network(combined)
