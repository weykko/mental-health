import torch
from torch.utils.data import Dataset, DataLoader


class DepressionDataset(Dataset):
    """
    Класс для создания кастомного датасета для задачи классификации депрессии и суицидальных мыслей.

    Параметры:
    texts (pandas.Series): Тексты сообщений.
    labels (pandas.Series): Метки классов (0 - не суицидальные, 1 - суицидальные).
    tokenizer (transformers.PreTrainedTokenizer): Токенизатор для обработки текста.
    max_length (int): Максимальная длина последовательности для токенизации.
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Возвращает количество примеров в датасете.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Получение одного примера из датасета, включая его текст, метку и токенизированное представление.

        Параметры:
        idx (int): Индекс примера в датасете.

        Возвращает:
        dict: Словарь с токенизированным текстом (input_ids), маской внимания (attention_mask) и меткой (label).
        """
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def get_depression_loaders(X_train, y_train, X_test, y_test, tokenizer, max_length, batch_size=64):
    """
    Создание загрузчиков данных для обучения и тестирования модели.

    Параметры:
    X_train (pandas.Series): Тексты обучающей выборки.
    y_train (pandas.Series): Метки классов обучающей выборки.
    X_test (pandas.Series): Тексты тестовой выборки.
    y_test (pandas.Series): Метки классов тестовой выборки.
    tokenizer (BertTokenizer): Токенизатор для преобразования текста в числовое представление.
    max_length (int): Максимальная длина токенизированных последовательностей.
    batch_size (int): Размер батча для загрузчиков данных.

    Возвращает:
    tuple: Кортеж, содержащий загрузчики данных для обучающей и тестовой выборок.
    """
    train_dataset = DepressionDataset(X_train, y_train, tokenizer, max_length)
    test_dataset = DepressionDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
