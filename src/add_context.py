""" add context to json files"""

import json
import copy
from datetime import datetime
from typing import Optional, Dict, Any


def add_context_to_json(file_path: str, character: str, new_context: str) -> bool:
    """
    Находит первое сообщение указанного персонажа в JSON файле и создает новое сообщение
    на его основе с измененным контекстом.
    
    :param file_path: Путь к JSON файлу
    :param character: Имя персонажа для поиска
    :param new_context: Новый контекст для создаваемого сообщения
    :return: True если операция успешна, False иначе
    """
    try:
        # Читаем JSON файл
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Поддерживаем разные структуры JSON
        messages = []
        if isinstance(data, dict) and 'messages' in data:
            messages = data['messages']
        elif isinstance(data, list):
            messages = data
        else:
            print(f"Неподдерживаемая структура JSON в файле {file_path}")
            return False
        
        # Находим первое сообщение указанного персонажа
        base_message = None
        for message in messages:
            if isinstance(message, dict) and message.get('character') == character:
                base_message = message
                break
        
        if not base_message:
            print(f"Персонаж '{character}' не найден в файле {file_path}")
            return False
        
        # Создаем новое сообщение на основе найденного
        new_message = copy.deepcopy(base_message)
        
        # Обновляем контекст и другие поля
        new_message['context'] = new_context
        new_message['timestamp'] = datetime.now().isoformat() + 'Z'
        
        # Генерируем новый ID
        existing_ids = [msg.get('id', '') for msg in messages if isinstance(msg, dict)]
        new_id = generate_new_id(existing_ids)
        new_message['id'] = new_id
        
        # Добавляем пометку о том, что это сгенерированное сообщение
        # new_message['generated'] = True
        # new_message['based_on'] = base_message.get('id', 'unknown')
        
        # Добавляем новое сообщение в конец
        messages.append(new_message)
        
        # Сохраняем обновленный JSON
        if isinstance(data, dict) and 'messages' in data:
            data['messages'] = messages
        else:
            data = messages
            
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        
        print(f"Новое сообщение для персонажа '{character}' с контекстом '{new_context}' добавлено в {file_path}")
        print(f"Базовое сообщение: {base_message.get('content', '')[:100]}...")
        print(f"Новый ID: {new_id}")
        
        return True
    
    except FileNotFoundError:
        print(f"Файл {file_path} не найден")
        return False
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON в файле {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Ошибка при добавлении контекста в {file_path}: {e}")
        return False


def generate_new_id(existing_ids: list) -> str:
    """
    Генерирует новый уникальный ID для сообщения
    
    :param existing_ids: Список существующих ID
    :return: Новый уникальный ID
    """
    # Пытаемся найти числовые ID и взять следующий
    max_num = 0
    for id_str in existing_ids:
        if id_str.startswith('msg_'):
            try:
                num = int(id_str.split('_')[1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue
    
    return f"msg_{max_num + 1:03d}"


def find_character_message(file_path: str, character: str) -> Optional[Dict[Any, Any]]:
    """
    Находит первое сообщение указанного персонажа в JSON файле
    
    :param file_path: Путь к JSON файлу
    :param character: Имя персонажа для поиска
    :return: Словарь с сообщением или None если не найдено
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        messages = []
        if isinstance(data, dict) and 'messages' in data:
            messages = data['messages']
        elif isinstance(data, list):
            messages = data
        
        for message in messages:
            if isinstance(message, dict) and message.get('character') == character:
                return message
        
        return None
    
    except Exception as e:
        print(f"Ошибка при поиске сообщения персонажа: {e}")
        return None


def preview_character_messages(file_path: str, character: str, limit: int = 5) -> list:
    """
    Показывает предварительный просмотр сообщений персонажа
    
    :param file_path: Путь к JSON файлу
    :param character: Имя персонажа
    :param limit: Максимальное количество сообщений для показа
    :return: Список сообщений персонажа
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        messages = []
        if isinstance(data, dict) and 'messages' in data:
            messages = data['messages']
        elif isinstance(data, list):
            messages = data
        
        character_messages = []
        for message in messages:
            if isinstance(message, dict) and message.get('character') == character:
                character_messages.append(message)
                if len(character_messages) >= limit:
                    break
        
        return character_messages
    
    except Exception as e:
        print(f"Ошибка при предварительном просмотре: {e}")
        return []


def batch_add_context(file_paths: list, character: str, new_context: str) -> Dict[str, bool]:
    """
    Добавляет контекст для персонажа в несколько файлов
    
    :param file_paths: Список путей к JSON файлам
    :param character: Имя персонажа
    :param new_context: Новый контекст
    :return: Словарь с результатами для каждого файла
    """
    results = {}
    
    for file_path in file_paths:
        print(f"\n--- Обработка файла: {file_path} ---")
        
        # Предварительный просмотр
        preview = preview_character_messages(file_path, character, 1)
        if preview:
            print(f"Найдено сообщение от {character}:")
            print(f"  Контекст: {preview[0].get('context', 'Не указан')}")
            print(f"  Содержание: {preview[0].get('content', '')[:100]}...")
        
        # Добавляем контекст
        results[file_path] = add_context_to_json(file_path, character, new_context)
    
    return results


if __name__ == "__main__":
    # Пример использования
    json_files = [
        'forum_knowledge_base/users.json',
        # 'forum_knowledge_base/alaev_messages.json',
        # 'forum_knowledge_base/senior_dev_messages.json',
        # Добавьте больше JSON файлов по необходимости
    ]
    
    character = "Alaev"
    new_context = """
Есть общепризнанная терминология. Есть принципиальные различия между роботом, механикой и автоматом.
Робот и автомат тебе никогда не дадут полного контроля над управляемостью авто.
Тебе это не нужно. Тебе и редуктор сойдет. 
У тебя колеса авто летают над ямами по баллистическим траекториям 🤣
"""
    
    print(f"🔍 Поиск сообщений персонажа '{character}' и добавление нового контекста...")
    print(f"📝 Новый контекст: '{new_context}'")
    print("=" * 70)
    
    # Пакетная обработка файлов
    results = batch_add_context(json_files, character, new_context)
    
    # Показываем результаты
    print("\n" + "=" * 70)
    print("📊 РЕЗУЛЬТАТЫ:")
    print("=" * 70)
    
    success_count = 0
    for file_path, success in results.items():
        status = "✅ УСПЕШНО" if success else "❌ ОШИБКА"
        print(f"{status}: {file_path}")
        if success:
            success_count += 1
    
    print(f"\n🎯 Обработано успешно: {success_count}/{len(json_files)} файлов")
    
    # Демонстрация других функций
    print("\n" + "=" * 70)
    print("🔍 ПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР:")
    print("=" * 70)
    
    for file_path in json_files:
        print(f"\n--- Файл: {file_path} ---")
        messages = preview_character_messages(file_path, character, 3)
        
        if messages:
            print(f"Найдено {len(messages)} сообщений от {character}:")
            for i, msg in enumerate(messages, 1):
                print(f"  {i}. [{msg.get('context', 'Без контекста')}] {msg.get('content', '')[:80]}...")
        else:
            print(f"Сообщения от {character} не найдены")