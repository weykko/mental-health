import torch
from transformers import BertForSequenceClassification, BertTokenizer
from utils.utils import preprocess_text


class DepressionPredictionModel:
    """
    Класс для работы с обученной моделью BERT для предсказания наличия депрессии или суицидальных мыслей в тексте.

    Этот класс загружает обученную модель и токенизатор, а также предоставляет метод для получения предсказания
    на основе текста.

    Параметры:
    model_path (str): Путь к директории с обученной моделью и токенизатором.
    device (str): Устройство для выполнения вычислений (по умолчанию "cpu").
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Инициализация класса, загрузка модели и токенизатора.

        Параметры:
        model_path (str): Путь к директории с обученной моделью.
        device (str): Устройство для выполнения вычислений ("cpu" или "cuda").
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.max_length = 128  # Максимальная длина для токенизации
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        """
        Загрузка модели и токенизатора из указанного пути.

        Возвращает:
        tuple: Содержит модель и токенизатор.
        """
        model = BertForSequenceClassification.from_pretrained(self.model_path)
        tokenizer = BertTokenizer.from_pretrained(self.model_path)
        model.to(self.device)  # Переводим модель на выбранное устройство
        return model, tokenizer

    def get_prediction(self, text: str):
        """
        Получение предсказания для переданного текста, используя обученную модель.

        Параметры:
        text (str): Входной текст для предсказания.

        Возвращает:
        float: Вероятность того, что текст содержит признаки суицида (значение от 0 до 1).
        """
        # Предобработка текста
        processed_text = preprocess_text(text)

        # Токенизация текста
        encoding = self.tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Переводим данные на устройство (GPU или CPU)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        self.model.eval()  # Переводим модель в режим оценки

        # Выполнение инференса (без вычисления градиентов)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Получаем предсказание
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Возвращаем вероятность наличия суицида (второй класс)
        return probs[0][1].item()
