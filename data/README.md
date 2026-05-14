# Данные

Положите сюда CSV-файлы датасета:

| Файл | Описание |
|---|---|
| `detailed_data.csv` | Полный датасет (25 АЗС × 365 дней × 24 ч, 219 000 строк) |
| `5stations_data.csv` | Срез из 5 АЗС для быстрого тестирования (43 800 строк) |
| `stations_metadata.csv` | Статические характеристики 25 АЗС |
| `5stations_metadata.csv` | Метаданные для среза из 5 АЗС |

Сами CSV-файлы не попадают в git (исключены через `.gitignore`).

## Sample dataset

В папке `data/sample/` лежит **маленький обезличенный пример** (2 АЗС × 14 дней,
672 строки) — он включён в git, чтобы проект можно было запустить сразу после
клонирования без полного датасета. Имена АЗС обезличены (`Station_A`,
`Station_B`).

Запуск на sample:

```powershell
$env:TABD_DATA_DIR = "data/sample"
python run.py --mode train --quick
```

Регенерация sample из полного датасета:

```powershell
python scripts/make_sample.py --source "C:/path/to/full/csv"
```

## Альтернативное расположение

Если CSV-файлы лежат в другой папке, укажите путь через переменную окружения
`TABD_DATA_DIR` перед запуском:

```powershell
$env:TABD_DATA_DIR = "D:\path\to\data"
python run.py --mode train
```

## Дашборд без CSV

После первого обучения данные кэшируются в `models/data_cache.parquet`.
Дашборд работает напрямую из кэша — CSV-файлы для него уже не нужны.
