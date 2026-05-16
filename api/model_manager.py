from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from utils import DataPreprocessor, DenseNN, FeatureExtractor
from nltk import word_tokenize
from config import settings


class ModelManager():
    def __init__(self) -> None:
        self.spam_model = None
        self.toxic_model = None
        self.toxic_tokenizer = None
        self.word2ind = None
        self.feature_extractor = FeatureExtractor()
        self.data_preprocessor = DataPreprocessor()
        self.features_col = ['text_length', 'word_count', 'digit_count', 'special_char_count', 
                           'has_url', 'has_phone_number', 'has_email']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_toxic_model(self):
        """Загрузка модели классификации на токсичность"""
        self.toxic_model = AutoModelForSequenceClassification.from_pretrained(settings.toxic_model_path)
        self.toxic_tokenizer = AutoTokenizer.from_pretrained(settings.toxic_model_path)
        self.toxic_model.to(self.device)
        self.toxic_model.eval()
        
    def load_spam_model(self):
        """Загрузка модели классификации на спам"""
        checkpoint = torch.load(settings.spam_model_path, map_location=self.device)
        self.spam_model = DenseNN(**checkpoint['model_config'])
        self.spam_model.load_state_dict(checkpoint['model_state_dict'])
        self.spam_model.to(self.device)
        self.spam_model.eval()
        self.word2ind = checkpoint['word2ind']
    
    def toxic_predict(self, text:str) -> float:
        """Предсказание вероятности токсичности текста с помощью загруженной модели"""
        if self.toxic_model is None:
            raise ValueError("Модель(toxic) не загружена")
        
        text = self.data_preprocessor.all_preprocess(text=text, stopwords=False, light_preprocess=True)
        inputs = self.toxic_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.toxic_model(**inputs)
        prediction = torch.sigmoid(outputs.logits).cpu().numpy()
        return prediction[0][1]
    
    def _prepare_sample(self, text) -> tuple[torch.Tensor,torch.Tensor]:
        """
        Приватный метод подготовки одного примера для инференса

        Выполняет полную preprocessing pipeline:
            - Извлечение признаков
            - Предобработку текста
            - Токенизацию с добавлением специальных токенов
            - Создание тензоров

        Возвращает тензоры текста и признаков, перенесенные на устройство
        """
        if self.word2ind is None:
            raise ValueError("Словарь(word2ind) не загружен")
        
        bos_id = self.word2ind['<bos>']
        eos_id = self.word2ind['<eos>']
        unk_id = self.word2ind['<unk>']

        features = self.feature_extractor.extract_all_features(text)
        text = self.data_preprocessor.all_preprocess(text, light_preprocess=True, stopwords=False)
        
        tokenized_msg = [bos_id]
        tokenized_msg += [self.word2ind.get(word, unk_id) for word in word_tokenize(text, language='russian', preserve_line=True)]
        tokenized_msg += [eos_id]
        
        text_tensor = torch.LongTensor(tokenized_msg).unsqueeze(0).to(self.device)
        features_list = [features[name] for name in self.features_col]
        features_tensor = torch.FloatTensor(features_list).unsqueeze(0).to(self.device)
            
        return text_tensor, features_tensor
    
    def spam_predict(self, text:str) -> float:
        """Предсказание вероятности спама текста с помощью загруженной модели"""
        if self.spam_model is None:
            raise ValueError('Спам модель не загружена')
        
        text_tensor, features_tensor = self._prepare_sample(text=text)
        
        with torch.no_grad():
            logits = self.spam_model(text_x = text_tensor,features_x = features_tensor)
            prediction = torch.sigmoid(logits).cpu().numpy()
        
        return prediction[0][0]
    
model_manager = ModelManager()